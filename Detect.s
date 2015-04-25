; yasm -f elf64 -Worphan-labels Detect.s
; gcc -nostdlib Detect.o -o Detect

; Returns:  has_avx | has_fma | has_avx2 | has_xmm_ymm_enabled
%define has_avx 1
%define has_fma 2
%define has_avx2 4
%define has_xmm_ymm_enabled 8


; Feature bits for masking, see e.g.:
; https://en.wikipedia.org/wiki/CPUID#EAX.3D1:_Processor_Info_and_Feature_Bits

; eax=1, ecx:  1 << 28 | 1 << 27
%define avx_mask 0x18000000

; eax=1, ecx:  1 << 28 | 1 << 27 | 1 << 12
%define fma_mask 0x18001000

; eax=7, ecx=0, ebx:  1 << 5
%define avx2_mask 0x20

; both xmm and ymm state enabled by OS
%define xmm_ymm_enabled 0x6


section .text
  global _start

_start:
  ; rv
  mov rdi, 0

  mov eax, 1
  cpuid
  and ecx, avx_mask
  cmp ecx, avx_mask
  jne no_avx
  or rdi, has_avx

  mov eax, 1
  cpuid
  and ecx, fma_mask
  cmp ecx, fma_mask
  jne no_fma
  or rdi, has_fma

  mov eax, 7
  mov ecx, 0
  cpuid
  and ebx, avx2_mask
  cmp ebx, avx2_mask
  jne no_avx2
  or rdi, has_avx2

  mov ecx, 0
  xgetbv
  and eax, xmm_ymm_enabled
  cmp eax, xmm_ymm_enabled
  jne no_xmm_ymm_enabled
  or rdi, has_xmm_ymm_enabled

no_avx:
no_fma:
no_avx2:
no_xmm_ymm_enabled:
  ; sys_exit
  mov rax, 60
  syscall
