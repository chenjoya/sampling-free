// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// This file is modified from  https://github.com/pytorch/pytorch/blob/master/modules/detectron/sigmoid_focal_loss_op.cu
// Cheng-Yang Fu
// cyfu@cs.unc.edu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void CELossForward(const int nthreads, 
    const T* logits,
    const int* targets,
    const int num_classes,
    const int num, 
    T* losses) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    int n = i / num_classes;
    int d = i % num_classes; // current class[0~79]; 
    int t = targets[n]; // target class [1~80];

    // Decide it is positive or negative case. 
    T c1 = (t == (d+1)); 
    T c2 = (t>=0 & t != (d+1));

    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    T  p = 1. / (1. + expf(-logits[i]));

    // log(p) where
    T term1 = logf(max(p, FLT_MIN));

    // log(1-p)
    T term2 = (-1. * logits[i] * (logits[i] >= 0) -   
             logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1;
    losses[i] += -c2 * term2;

  } // CUDA_1D_KERNEL_LOOP
} // CELossForward


template <typename T>
__global__ void CELossBackward(const int nthreads,
                const T* logits,
                const int* targets,
                const T* d_losses,
                const int num_classes,
                const int num,
                T* d_logits) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {

    int n = i / num_classes;
    int d = i % num_classes; // current class[0~79]; 
    int t = targets[n]; // target class [1~80], 0 is background;

    // Decide it is positive or negative case. 
    T c1 = (t == (d+1));
    T c2 = (t>=0 & t != (d+1));

    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    T  p = 1. / (1. + expf(-logits[i]));

    // 1. - p
    T term1 = 1. - p;

    // -p
    T term2 = 0. - p;
    
    d_logits[i] = 0.0;
    d_logits[i] += -c1 * term1;
    d_logits[i] += -c2 * term2;
    d_logits[i] = d_logits[i] * d_losses[i];

  } // CUDA_1D_KERNEL_LOOP
} // CELossBackward


at::Tensor CELoss_forward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const int num_classes) {
  AT_ASSERTM(logits.type().is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.type().is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");

  const int num_samples = logits.size(0);
	
  auto losses = at::empty({num_samples, logits.size(1)}, logits.options());
  auto losses_size = num_samples * logits.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)losses_size, 512L), 4096L));
  dim3 block(512);

  if (losses.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return losses;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.type(), "CELoss_forward", [&] {
    CELossForward<scalar_t><<<grid, block, 0, stream>>>(
         losses_size,
         logits.contiguous().data<scalar_t>(),
	       targets.contiguous().data<int>(),
         num_classes,
	       num_samples,
         losses.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return losses;   
}	


at::Tensor CELoss_backward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const at::Tensor& d_losses,
		const int num_classes) {
  AT_ASSERTM(logits.type().is_cuda(), "logits must be a CUDA tensor");
  AT_ASSERTM(targets.type().is_cuda(), "targets must be a CUDA tensor");
  AT_ASSERTM(d_losses.type().is_cuda(), "d_losses must be a CUDA tensor");

  AT_ASSERTM(logits.dim() == 2, "logits should be NxClass");

  const int num_samples = logits.size(0);
  AT_ASSERTM(logits.size(1) == num_classes, "logits.size(1) should be num_classes");
	
  auto d_logits = at::zeros({num_samples, num_classes}, logits.options());
  auto d_logits_size = num_samples * logits.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)d_logits_size, 512L), 4096L));
  dim3 block(512);

  if (d_logits.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return d_logits;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.type(), "CELoss_backward", [&] {
    CELossBackward<scalar_t><<<grid, block, 0, stream>>>(
         d_logits_size,
         logits.contiguous().data<scalar_t>(),
	       targets.contiguous().data<int>(),
	       d_losses.contiguous().data<scalar_t>(),
         num_classes,
	       num_samples,
         d_logits.data<scalar_t>());
  });

  THCudaCheck(cudaGetLastError());
  return d_logits;   
}	

