%dx.types.Handle = type { i8* }
%struct.RWByteAddressBuffer = type { i32 }

@dx.v32.r0 = internal global i32 undef
@dx.v32.r1 = internal global i32 undef

define internal void @dx.label.0() {
entry:
  call void @dx.label.1()
  ret void
}

define internal void @dx.label.1() {
entry:
  %a = load i32, i32* @dx.v32.r0, align 4
  %m = mul i32 %a, 3
  %v = add i32 %m, 7
  store i32 %v, i32* @dx.v32.r1, align 4
  ret void
}

define void @main() {
entry:
  %h = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  %tid = call i32 @dx.op.threadId.i32(i32 93, i32 0)
  store i32 %tid, i32* @dx.v32.r0, align 4
  call void @dx.label.0()
  %v = load i32, i32* @dx.v32.r1, align 4
  %off = shl i32 %tid, 2
  call void @dx.op.bufferStore.i32(i32 69, %dx.types.Handle %h, i32 %off, i32 undef, i32 %v, i32 %v, i32 %v, i32 %v, i8 1)
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1) #2

; Function Attrs: nounwind
declare void @dx.op.bufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readonly }

!dx.version = !{!0}
!dx.valver = !{!0}
!dx.shaderModel = !{!1}
!dx.resources = !{!2}
!dx.entryPoints = !{!5}
!llvm.ident = !{!9}

!0 = !{i32 1, i32 0}
!1 = !{!"cs", i32 6, i32 0}
!2 = !{null, !3, null, null}
!3 = !{!4}
!4 = !{i32 0, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!5 = !{void ()* @main, !"main", null, !2, !6}
!6 = !{i32 0, i64 0, i32 4, !7}
!7 = !{i32 64, i32 1, i32 1}
!9 = !{!"dxbc2dxil 1.2"}
