// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Phase 2c lowering: ports the universal subset of FlyToROCDL into the xegpu
// backend so a backend-neutral kernel (e.g. vectorAdd) can reach upstream
// MLIR's `gpu-lower-to-xevm-pipeline`. We deliberately omit:
//   - BufferDesc* / RawPtrBuffer* (AMD-only buffer-resource path)
//   - MmaAtom / TiledMma / MmaAtomCall* (no xegpu MMA atoms yet; vectorAdd is
//     matmul-free)
//   - FlyROCDLClusterAttrPass (amdgpu-cluster-dims is rocdl-only)
//
// New xegpu copy / MMA atom types will plug in via the same
// Fly_{Copy,Mma}OpTypeInterface / Fly_StatefulOpTypeInterface contract used
// by FlyROCDL, but no concrete fly_xegpu types exist yet.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"

#include "flydsl/Conversion/FlyToXeGPU/FlyToXeGPU.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/FlyXeGPU/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_FLYTOXEGPUCONVERSIONPASS
#include "flydsl/Conversion/FlyToXeGPU/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly;

namespace {

// XeGPU lowers to SPIR-V via xevm. The relevant SPIR-V storage classes map
// onto LLVM address spaces as follows in the upstream `convert-gpu-to-llvm-spv`
// path: CrossWorkgroup=1 (Global), Workgroup=3 (Shared). Private memory in
// SPIR-V is StorageClass::Function=7, but the upstream lowering routes
// register / private allocas through the generic AS=0; we follow that
// convention here. Returning AS=0 for Register keeps vectorAdd correct since
// it does not stress private memory; revisit when adding kernels that do.
unsigned mapToLLVMAddressSpace(AddressSpace addrSpace) {
  switch (addrSpace) {
  case AddressSpace::Generic:
    return 0;
  case AddressSpace::Global:
    return 1;
  case AddressSpace::Shared:
    return 3;
  case AddressSpace::Register:
    return 0; // SPIR-V private (Function=7) collapsed onto generic for now
  }
  llvm_unreachable("unsupported address space");
}

// xegpu has no buffer-descriptor address space; any unknown attribute defaults
// to AS=0. (FlyToROCDL maps BufferDescAddressAttr -> 8.)
unsigned mapAttrToLLVMAddressSpace(Attribute attr) {
  if (auto e = dyn_cast<AddressSpaceAttr>(attr))
    return mapToLLVMAddressSpace(e.getValue());
  return 0;
}

// Create a freshly named LDS global of `[nbytes x i8]` in `addrSpace`, inserted
// at the start of `moduleOp`. The symbol name is `prefix` followed by a counter
// chosen to avoid collisions with existing globals in the module. Shared by
// the static-LDS `make_ptr` and dynamic-LDS `get_dyn_shared` lowerings.
static LLVM::GlobalOp createLDSGlobal(ConversionPatternRewriter &rewriter,
                                      gpu::GPUModuleOp moduleOp, Location loc, StringRef prefix,
                                      int64_t nbytes, int64_t align, unsigned addrSpace) {
  llvm::StringSet<> existingNames;
  for (auto globalOp : moduleOp.getBody()->getOps<LLVM::GlobalOp>())
    existingNames.insert(globalOp.getSymName());

  unsigned counter = 0;
  SmallString<128> symName = SymbolTable::generateSymbolName<128>(
      prefix, [&](StringRef candidate) { return existingNames.contains(candidate); }, counter);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto arrayTy = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), nbytes);
  auto globalOp = LLVM::GlobalOp::create(rewriter, loc, arrayTy,
                                         /*isConstant=*/false, LLVM::Linkage::External, symName,
                                         /*value=*/Attribute(),
                                         /*alignment=*/align, addrSpace);
  // Workgroup (addrspace 3) symbols are per-workgroup hardware and can never
  // resolve across DSO; same dso_local invariant as the AMD LDS path.
  globalOp.setDsoLocal(true);
  return globalOp;
}

class MakePtrOpLowering : public OpConversionPattern<MakePtrOp> {
public:
  MakePtrOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<MakePtrOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(MakePtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getResult().getType());
    if (!flyPtrTy)
      return failure();

    Location loc = op.getLoc();
    Attribute addrSpaceAttr = flyPtrTy.getAddressSpace();

    if (isGenericAddressSpace<AddressSpace::Register>(addrSpaceAttr)) {
      auto dictAttrs = op.getDictAttrs();
      if (!dictAttrs)
        return rewriter.notifyMatchFailure(op, "register make_ptr requires dictAttrs");
      auto allocSize = dictAttrs->getAs<IntegerAttr>("allocSize");
      if (!allocSize)
        return rewriter.notifyMatchFailure(op, "register make_ptr requires allocSize in ptrAttrs");
      unsigned llvmAS = mapToLLVMAddressSpace(AddressSpace::Register);
      auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), llvmAS);
      Value nElems = arith::ConstantIntOp::create(rewriter, loc, allocSize.getInt(), 64);
      Type elemTy = projectToLLVMCompatibleElemTy(flyPtrTy.getElemTy());
      Value ptr = LLVM::AllocaOp::create(rewriter, loc, llvmPtrTy, elemTy, nElems, 0);
      rewriter.replaceOp(op, ptr);
      return success();
    } else if (isGenericAddressSpace<AddressSpace::Shared>(addrSpaceAttr)) {
      // Static LDS sub-allocation. Each op lowers to a freshly named
      // `@__shared_alloc_<n>` global in addrspace 3 (SPIR-V Workgroup).
      auto dictAttrs = op.getDictAttrs();
      if (!dictAttrs)
        return rewriter.notifyMatchFailure(
            op, "shared make_ptr requires dictAttrs={allocBytes, allocAlign}");
      auto allocBytesAttr = dictAttrs->getAs<IntegerAttr>("allocBytes");
      auto allocAlignAttr = dictAttrs->getAs<IntegerAttr>("allocAlign");
      if (!allocBytesAttr || !allocAlignAttr)
        return rewriter.notifyMatchFailure(
            op, "shared make_ptr requires allocBytes, allocAlign in dictAttrs");

      auto moduleOp = op->getParentOfType<gpu::GPUModuleOp>();
      if (!moduleOp)
        return op->emitError("shared make_ptr must be inside a gpu.module");

      LLVM::GlobalOp global =
          createLDSGlobal(rewriter, moduleOp, loc, "__shared_alloc", allocBytesAttr.getInt(),
                          allocAlignAttr.getInt(), /*addrSpace=*/3);
      rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(op, global);
      return success();
    }

    // BufferDesc address space is rocdl-only and intentionally unsupported here.
    return rewriter.notifyMatchFailure(op, "unsupported make_ptr address space for xegpu");
  }
};

class GetDynSharedOpLowering : public OpConversionPattern<GetDynSharedOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GetDynSharedOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto flyPtrTy = cast<fly::PointerType>(op.getResult().getType());
    unsigned addrSpace = mapAttrToLLVMAddressSpace(flyPtrTy.getAddressSpace());

    auto moduleOp = op->getParentOfType<gpu::GPUModuleOp>();
    if (!moduleOp)
      return op->emitError("get_dyn_shared must be inside a gpu.module");

    // Dynamic shared memory has a single logical region per kernel; reuse an
    // existing `[0 x i8]` global if one is already present so multiple
    // `get_dyn_shared` ops resolve to the same base symbol.
    LLVM::GlobalOp sharedGlobal;
    for (auto globalOp : moduleOp.getBody()->getOps<LLVM::GlobalOp>()) {
      if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(globalOp.getType())) {
        if (globalOp.getAddrSpace() == addrSpace && arrayType.getNumElements() == 0 &&
            globalOp.getAlignment().value_or(0) == 1024) {
          sharedGlobal = globalOp;
          break;
        }
      }
    }
    if (!sharedGlobal) {
      sharedGlobal = createLDSGlobal(rewriter, moduleOp, loc, "__dynamic_shared",
                                     /*nbytes=*/0, /*align=*/1024, addrSpace);
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto basePtr = LLVM::AddressOfOp::create(rewriter, loc, sharedGlobal);
    Type ptrType = basePtr->getResultTypes()[0];

    auto i8Ty = IntegerType::get(rewriter.getContext(), 8);
    Value sharedPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrType, i8Ty, basePtr, ArrayRef<LLVM::GEPArg>{0});

    rewriter.replaceOp(op, sharedPtr);
    return success();
  }
};

class IntToPtrOpLowering : public OpConversionPattern<IntToPtrOp> {
public:
  IntToPtrOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<IntToPtrOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(IntToPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getResult().getType());
    if (!flyPtrTy)
      return failure();

    auto resultTy = dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(flyPtrTy));
    if (!resultTy)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, resultTy, adaptor.getSrc());
    return success();
  }
};

class PtrToIntOpLowering : public OpConversionPattern<PtrToIntOp> {
public:
  PtrToIntOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<PtrToIntOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(PtrToIntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resultTy, adaptor.getPtr());
    return success();
  }
};

class ApplySwizzleOpLowering : public OpConversionPattern<ApplySwizzleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ApplySwizzleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

class RecastIterOpLowering : public OpConversionPattern<RecastIterOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(RecastIterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class AddOffsetOpLowering : public OpConversionPattern<AddOffsetOp> {
public:
  AddOffsetOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<AddOffsetOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(AddOffsetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value base = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    auto offsetTy = dyn_cast<fly::IntTupleType>(offset.getType());
    IntTupleAttr offsetAttr = offsetTy.getAttr();
    if (!offsetAttr.isLeaf())
      return rewriter.notifyMatchFailure(op, "offset must be a leaf int tuple");

    Value offsetVal;
    auto offsetInt = offsetAttr.extractIntFromLeaf();
    if (offsetInt.isStatic()) {
      offsetVal = arith::ConstantIntOp::create(rewriter, loc, offsetInt.getValue(), 32);
    } else {
      Operation *defOp = offset.getDefiningOp();
      offsetVal = defOp->getOperand(0);
    }

    // BufferDesc fat-pointer GEP is rocdl-only; xegpu uses generic GEP.
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(base.getType());
    if (!ptrTy)
      return failure();

    Type elemTy = projectToLLVMCompatibleElemTy(flyPtrTy.getElemTy());
    Value gep = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, base, ValueRange{offsetVal});
    rewriter.replaceOp(op, gep);
    return success();
  }
};

class MakeViewOpLowering : public OpConversionPattern<MakeViewOp> {
public:
  MakeViewOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<MakeViewOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(MakeViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (isa<fly::CoordTensorType>(op.getResult().getType())) {
      if (!op.getResult().use_empty())
        return rewriter.notifyMatchFailure(op, "coord_tensor result should have no uses");
      rewriter.eraseOp(op);
      return success();
    } else {
      Value base = adaptor.getIter();
      rewriter.replaceOp(op, base);
      return success();
    }
  }
};

class PtrLoadOpLowering : public OpConversionPattern<PtrLoadOp> {
public:
  using OpConversionPattern<PtrLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = adaptor.getPtr();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    Type loadTy = op.getResult().getType();

    if (auto vecTy = dyn_cast<VectorType>(loadTy)) {
      auto swizzle = flyPtrTy.getSwizzle();
      if (!swizzle.isTrivialSwizzle()) {
        int64_t vecBytes =
            vecTy.getNumElements() * vecTy.getElementType().getIntOrFloatBitWidth() / 8;
        int64_t baseBytes = int64_t{1} << swizzle.getBase();
        if (baseBytes % vecBytes != 0)
          return rewriter.notifyMatchFailure(
              op, "vector ptr.load byte size must divide swizzle base granularity");
      }
    }

    // Generic LLVM load only; the rocdl raw_ptr_buffer_load path is dropped.
    ptr = applySwizzleOnPtr(rewriter, loc, cast<TypedValue<LLVM::LLVMPointerType>>(ptr),
                            flyPtrTy.getSwizzle());
    Value loaded = LLVM::LoadOp::create(rewriter, loc, loadTy, ptr);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class PtrStoreOpLowering : public OpConversionPattern<PtrStoreOp> {
public:
  using OpConversionPattern<PtrStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Value value = adaptor.getValue();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    if (auto vecTy = dyn_cast<VectorType>(value.getType())) {
      auto swizzle = flyPtrTy.getSwizzle();
      if (!swizzle.isTrivialSwizzle()) {
        int64_t vecBytes =
            vecTy.getNumElements() * vecTy.getElementType().getIntOrFloatBitWidth() / 8;
        int64_t baseBytes = int64_t{1} << swizzle.getBase();
        if (baseBytes % vecBytes != 0)
          return rewriter.notifyMatchFailure(
              op, "vector ptr.store byte size must divide swizzle base granularity");
      }
    }

    // Generic LLVM store only; the rocdl raw_ptr_buffer_store path is dropped.
    ptr = applySwizzleOnPtr(rewriter, loc, cast<TypedValue<LLVM::LLVMPointerType>>(ptr),
                            flyPtrTy.getSwizzle());
    LLVM::StoreOp::create(rewriter, loc, value, ptr);
    rewriter.eraseOp(op);
    return success();
  }
};

class MakeCopyAtomOpLowering : public OpConversionPattern<MakeCopyAtomOp> {
public:
  using OpConversionPattern<MakeCopyAtomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MakeCopyAtomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto copyAtomTy = dyn_cast<CopyAtomType>(op.getResult().getType());
    if (!copyAtomTy)
      return rewriter.notifyMatchFailure(op, "not a CopyAtomType");
    Type convertedTy = getTypeConverter()->convertType(copyAtomTy);

    auto statefulOp = dyn_cast<StatefulOpTypeInterface>(copyAtomTy.getCopyOp());
    if (statefulOp) {
      Value state = statefulOp.getDefaultState(rewriter, op.getLoc());
      rewriter.replaceOp(op, state);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, convertedTy);
    }
    return success();
  }
};

class MakeTiledCopyOpLowering : public OpConversionPattern<MakeTiledCopyOp> {
public:
  using OpConversionPattern<MakeTiledCopyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MakeTiledCopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getCopyAtom());
    return success();
  }
};

class AtomSetValueOpLowering : public OpConversionPattern<AtomSetValueOp> {
public:
  using OpConversionPattern<AtomSetValueOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AtomSetValueOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type origAtomTy = op.getAtom().getType();
    StringAttr fieldAttr = op.getFieldAttr();
    Location loc = op.getLoc();

    Value structVal = adaptor.getAtom();
    Value fieldVal = adaptor.getValue();
    Value result;

    if (auto copyAtomTy = dyn_cast<CopyAtomType>(origAtomTy)) {
      if (!copyAtomTy.isStateful())
        return rewriter.notifyMatchFailure(op, "CopyAtom is not stateful");
      result = copyAtomTy.setAtomState(rewriter, loc, structVal, fieldAttr, fieldVal);
    } else if (auto mmaAtomTy = dyn_cast<MmaAtomType>(origAtomTy)) {
      if (!mmaAtomTy.isStateful())
        return rewriter.notifyMatchFailure(op, "MmaAtom is not stateful");
      result = mmaAtomTy.setAtomState(rewriter, loc, structVal, fieldAttr, fieldVal);
    } else {
      return rewriter.notifyMatchFailure(op, "atom is not CopyAtomType or MmaAtomType");
    }

    if (!result)
      return rewriter.notifyMatchFailure(op, "setAtomState failed");

    rewriter.replaceOp(op, result);
    return success();
  }
};

class CopyAtomCallLowering : public OpConversionPattern<CopyAtomCall> {
public:
  using OpConversionPattern<CopyAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(CopyAtomCall op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type copyAtomType = op.getCopyAtom().getType();
    auto copyAtom = dyn_cast<CopyAtomType>(copyAtomType);
    if (!copyAtom)
      return rewriter.notifyMatchFailure(op, "copyAtom is not CopyAtomType");

    Value copyAtomVal = adaptor.getCopyAtom();
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value pred = adaptor.getPred();

    auto srcMemTy = dyn_cast<fly::MemRefType>(op.getSrc().getType());
    auto dstMemTy = dyn_cast<fly::MemRefType>(op.getDst().getType());

    if (!srcMemTy || !dstMemTy)
      return rewriter.notifyMatchFailure(op, "expected MemRef types on original op");
    if (srcMemTy.getElemTy() != dstMemTy.getElemTy())
      return rewriter.notifyMatchFailure(op, "src/dst element types mismatch");

    Location loc = op.getLoc();

    Type predMemTy = nullptr;
    if (pred) {
      predMemTy = dyn_cast<fly::MemRefType>(op.getPred().getType());
      if (!predMemTy)
        return rewriter.notifyMatchFailure(op, "pred is not a MemRef type");
    }

    if (pred) {
      if (failed(copyAtom.emitAtomCall(rewriter, loc, copyAtomType, srcMemTy, dstMemTy, predMemTy,
                                       copyAtomVal, src, dst, pred)))
        return failure();
    } else {
      if (failed(copyAtom.emitAtomCall(rewriter, loc, copyAtomType, srcMemTy, dstMemTy, copyAtomVal,
                                       src, dst)))
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class CopyAtomCallSSALowering : public OpConversionPattern<CopyAtomCallSSA> {
public:
  using OpConversionPattern<CopyAtomCallSSA>::OpConversionPattern;

  LogicalResult matchAndRewrite(CopyAtomCallSSA op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type copyAtomType = op.getCopyAtom().getType();
    auto copyAtom = dyn_cast<CopyAtomType>(copyAtomType);
    if (!copyAtom)
      return rewriter.notifyMatchFailure(op, "copyAtom is not CopyAtomType");

    Location loc = op.getLoc();
    bool hasResult = op.getResults().size() > 0;
    Type srcTy = op.getSrc().getType();
    Value pred = adaptor.getPred();

    Type resultTy = hasResult ? op.getResult(0).getType() : Type{};
    Type dstTy = op.getDst() ? op.getDst().getType() : Type{};

    FailureOr<Value> result;
    if (pred) {
      result = copyAtom.emitAtomCallSSA(rewriter, loc, resultTy, copyAtomType, srcTy, dstTy,
                                        op.getPred().getType(), adaptor.getCopyAtom(),
                                        adaptor.getSrc(), adaptor.getDst(), pred);
    } else {
      result = copyAtom.emitAtomCallSSA(rewriter, loc, resultTy, copyAtomType, srcTy, dstTy,
                                        adaptor.getCopyAtom(), adaptor.getSrc(), adaptor.getDst());
    }
    if (failed(result))
      return failure();

    if (hasResult)
      rewriter.replaceOp(op, *result);
    else
      rewriter.eraseOp(op);
    return success();
  }
};

/// Lower `gpu.launch_func` kernel operands so that any `!fly.memref` values are
/// replaced by their type-converted builtin `memref` values. This prevents
/// `unrealized_conversion_cast` materializations from remaining live after
/// partial conversion (e.g., when the surrounding `func.func` signature has
/// been converted to builtin memrefs).
class GpuLaunchFuncOpLowering : public OpConversionPattern<gpu::LaunchFuncOp> {
public:
  using OpConversionPattern<gpu::LaunchFuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kernelRef = adaptor.getKernel();

    auto grid =
        gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
    auto block =
        gpu::KernelDim3{adaptor.getBlockSizeX(), adaptor.getBlockSizeY(), adaptor.getBlockSizeZ()};

    std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
    if (adaptor.getClusterSizeX() && adaptor.getClusterSizeY() && adaptor.getClusterSizeZ()) {
      clusterSize = gpu::KernelDim3{adaptor.getClusterSizeX(), adaptor.getClusterSizeY(),
                                    adaptor.getClusterSizeZ()};
    }

    // Preserve async token result type when present.
    Type asyncTokenType = nullptr;
    if (Value tok = op.getAsyncToken())
      asyncTokenType = tok.getType();

    // There are two relevant builder signatures in this MLIR:
    // - (kernel, ..., asyncTokenType, asyncDependencies, clusterSize)
    // - (kernel, ..., asyncObject, clusterSize)
    // Pick the one that matches the original op structure.
    if (Value asyncObj = adaptor.getAsyncObject()) {
      if (!adaptor.getAsyncDependencies().empty())
        return rewriter.notifyMatchFailure(
            op, "launch_func has both asyncObject and asyncDependencies");

      rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
          op, kernelRef, grid, block, adaptor.getDynamicSharedMemorySize(),
          adaptor.getKernelOperands(), asyncObj, clusterSize);
      return success();
    }

    rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
        op, kernelRef, grid, block, adaptor.getDynamicSharedMemorySize(),
        adaptor.getKernelOperands(), asyncTokenType, adaptor.getAsyncDependencies(), clusterSize);
    return success();
  }
};

class FlyTypeConverter : public TypeConverter {
public:
  FlyTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([&](FloatType floatTy) -> std::optional<Type> {
      if (floatTy.getWidth() < 16)
        return IntegerType::get(floatTy.getContext(), floatTy.getWidth());
      return std::nullopt;
    });
    addConversion([&](VectorType vecTy) -> std::optional<Type> {
      Type convertedElem = convertType(vecTy.getElementType());
      if (!convertedElem || convertedElem == vecTy.getElementType())
        return std::nullopt;
      return VectorType::get(vecTy.getShape(), convertedElem, vecTy.getScalableDims());
    });
    // No BufferDescAddressAttr branch: xegpu has no fat-pointer address space.
    addConversion([&](fly::MemRefType flyMemRefTy) -> Type {
      unsigned as = mapAttrToLLVMAddressSpace(flyMemRefTy.getAddressSpace());
      return LLVM::LLVMPointerType::get(flyMemRefTy.getContext(), as);
    });
    addConversion([&](fly::PointerType flyPtrTy) -> Type {
      unsigned as = mapAttrToLLVMAddressSpace(flyPtrTy.getAddressSpace());
      return LLVM::LLVMPointerType::get(flyPtrTy.getContext(), as);
    });
    addConversion([&](fly::CopyAtomType atomTy) -> Type {
      if (atomTy.isStateful())
        return atomTy.getConvertedType(atomTy.getContext());
      return LLVM::LLVMStructType::getLiteral(atomTy.getContext(), {});
    });
    addConversion([&](fly::MmaAtomType atomTy) -> Type {
      if (atomTy.isStateful())
        return atomTy.getConvertedType(atomTy.getContext());
      return LLVM::LLVMStructType::getLiteral(atomTy.getContext(), {});
    });
    addConversion(
        [&](fly::TiledCopyType tiledTy) -> Type { return convertType(tiledTy.getCopyAtom()); });
    addConversion(
        [&](fly::TiledMmaType tiledTy) -> Type { return convertType(tiledTy.getMmaAtom()); });
  }
};

class ExtractAlignedPointerAsIndexLowering
    : public OpConversionPattern<ExtractAlignedPointerAsIndexOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // fly.memref is a bare pointer; after type conversion the operand is llvm.ptr<AS>.
    // Cast to the result type (e.g. llvm.ptr<0>) if address spaces differ.
    Value src = adaptor.getSource();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      resultType = op.getResult().getType();
    if (src.getType() != resultType)
      src = LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), resultType, src);
    rewriter.replaceOp(op, src);
    return success();
  }
};

class FlyToXeGPUConversionPass
    : public mlir::impl::FlyToXeGPUConversionPassBase<FlyToXeGPUConversionPass> {
public:
  using mlir::impl::FlyToXeGPUConversionPassBase<
      FlyToXeGPUConversionPass>::FlyToXeGPUConversionPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect,
                           gpu::GPUDialect, func::FuncDialect, LLVM::LLVMDialect,
                           xegpu::XeGPUDialect>();
    target.addIllegalDialect<fly::FlyDialect>();

    // Constructors
    target.addLegalOp<StaticOp, MakeIntTupleOp, MakeLayoutOp, MakeComposedLayoutOp>();

    FlyTypeConverter typeConverter;

    // Ensure function signatures are type-converted; otherwise conversions may rely on
    // inserted unrealized casts that remain live.
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>(
        [&](gpu::GPUFuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });

    // IMPORTANT: `gpu.launch_func` itself is in a legal dialect, but its kernel operands may
    // still carry illegal `!fly.memref` types. If we don't mark it dynamically illegal in that
    // case, partial conversion won't try to rewrite it, leaving `unrealized_conversion_cast`
    // users alive and causing legalization failure.
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>([&](gpu::LaunchFuncOp op) {
      auto isValueLegal = [&](Value v) {
        if (!v)
          return true;
        return typeConverter.isLegal(v.getType());
      };

      for (Value v : op.getKernelOperands())
        if (!isValueLegal(v))
          return false;

      if (!isValueLegal(op.getDynamicSharedMemorySize()))
        return false;

      // Async operands are part of the operand list; keep them consistent as well.
      for (Value dep : op.getAsyncDependencies())
        if (!isValueLegal(dep))
          return false;
      if (!isValueLegal(op.getAsyncObject()))
        return false;

      // Dimensions are typically index and already legal; no need to special-case.
      return true;
    });

    patterns.add<MakePtrOpLowering, GetDynSharedOpLowering>(typeConverter, context);
    patterns.add<IntToPtrOpLowering, PtrToIntOpLowering>(typeConverter, context);
    patterns.add<ApplySwizzleOpLowering, RecastIterOpLowering>(typeConverter, context);
    patterns.add<AddOffsetOpLowering>(typeConverter, context);
    patterns.add<MakeViewOpLowering>(typeConverter, context);
    patterns.add<PtrLoadOpLowering, PtrStoreOpLowering>(typeConverter, context);
    patterns.add<MakeCopyAtomOpLowering>(typeConverter, context);
    patterns.add<MakeTiledCopyOpLowering>(typeConverter, context);
    patterns.add<AtomSetValueOpLowering>(typeConverter, context);
    patterns.add<CopyAtomCallLowering>(typeConverter, context);
    patterns.add<CopyAtomCallSSALowering>(typeConverter, context);
    patterns.add<GpuLaunchFuncOpLowering>(typeConverter, context);

    // TODO: deprecated in the future
    patterns.add<ExtractAlignedPointerAsIndexLowering>(typeConverter, context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
