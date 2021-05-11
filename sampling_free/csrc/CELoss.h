#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor CELoss_forward(
		const at::Tensor& logits,
    const at::Tensor& targets,
		const int num_classes) {
  if (logits.type().is_cuda()) {
#ifdef WITH_CUDA
    return CELoss_forward_cuda(logits, targets, num_classes);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor CELoss_backward(
			     const at::Tensor& logits,
           const at::Tensor& targets,
			     const at::Tensor& d_losses,
			     const int num_classes) {
  if (logits.type().is_cuda()) {
#ifdef WITH_CUDA
    return CELoss_backward_cuda(logits, targets, d_losses, num_classes);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
