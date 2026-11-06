@dx.v32.r0 = internal global float undef
@dx.v32.r1 = internal global float undef
@dx.v32.r2 = internal global float undef
@dx.v32.r3 = internal global float undef
@dx.v32.r4 = internal global float undef
@dx.v32.r5 = internal global float undef

define internal void @dx.label.0(i32 %v) {
entry:
  %p0 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 0, i32 %v)
  store float %p0, float* @dx.v32.r0, align 4
  %p1 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 1, i32 %v)
  store float %p1, float* @dx.v32.r1, align 4
  %p2 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 2, i32 %v)
  store float %p2, float* @dx.v32.r2, align 4
  %p3 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 3, i32 %v)
  store float %p3, float* @dx.v32.r3, align 4
  %t0 = call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 0, i32 %v)
  store float %t0, float* @dx.v32.r4, align 4
  %t1 = call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 1, i32 %v)
  store float %t1, float* @dx.v32.r5, align 4
  call void @dx.label.1()
  ret void
}

define internal void @dx.label.1() {
entry:
  %p0 = load float, float* @dx.v32.r0, align 4
  %p1 = load float, float* @dx.v32.r1, align 4
  %p2 = load float, float* @dx.v32.r2, align 4
  %p3 = load float, float* @dx.v32.r3, align 4
  %t0 = load float, float* @dx.v32.r4, align 4
  %t1 = load float, float* @dx.v32.r5, align 4
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 0, float %p0)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 1, float %p1)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 2, float %p2)
  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 3, float %p3)
  call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 0, float %t0)
  call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 1, float %t1)
  call void @dx.op.emitStream(i32 97, i8 0)
  ret void
}

define void @main() {
entry:
  call void @dx.label.0(i32 0)
  call void @dx.label.0(i32 1)
  call void @dx.label.0(i32 2)
  call void @dx.op.cutStream(i32 98, i8 0)
  ret void
}

; Function Attrs: nounwind
declare void @dx.op.storeOutput.f32(i32, i32, i32, i8, float) #0

; Function Attrs: nounwind readnone
declare float @dx.op.loadInput.f32(i32, i32, i32, i8, i32) #1

; Function Attrs: nounwind
declare void @dx.op.emitStream(i32, i8) #0

; Function Attrs: nounwind
declare void @dx.op.cutStream(i32, i8) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!dx.version = !{!0}
!dx.valver = !{!0}
!dx.shaderModel = !{!1}
!dx.entryPoints = !{!2}
!llvm.ident = !{!14}

!0 = !{i32 1, i32 0}
!1 = !{!"gs", i32 6, i32 0}
!2 = !{void ()* @main, !"main", !3, null, !11}
!3 = !{!4, !8, null}
!4 = !{!5, !7}
!5 = !{i32 0, !"SV_Position", i8 9, i8 3, !6, i8 4, i32 1, i8 4, i32 0, i8 0, null}
!6 = !{i32 0}
!7 = !{i32 1, !"TEXCOORD", i8 9, i8 0, !6, i8 2, i32 1, i8 2, i32 1, i8 0, null}
!8 = !{!9, !10}
!9 = !{i32 0, !"SV_Position", i8 9, i8 3, !6, i8 4, i32 1, i8 4, i32 0, i8 0, null}
!10 = !{i32 1, !"TEXCOORD", i8 9, i8 0, !6, i8 2, i32 1, i8 2, i32 1, i8 0, null}
!11 = !{i32 0, i64 0, i32 1, !12}
!12 = !{i32 3, i32 3, i32 1, i32 5, i32 1}
!14 = !{!"dxbc2dxil 1.2"}
