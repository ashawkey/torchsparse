#include <torch/torch.h>
#include <vector>
#include "devox_gpu.h"


//make sure indices is int type
at::Tensor deterministic_devoxelize_forward(
    const at::Tensor feat,     // [N, f]
    const at::Tensor indices,  // [M, 8]
    const at::Tensor weight    // [M, 8]
){  
  int M = indices.size(0);
  int c = feat.size(1);
  int l = weight.size(1);
  
  at::Tensor out = torch::zeros({M, c}, at::device(feat.device()).dtype(at::ScalarType::Float));
  deterministic_devoxelize_wrapper(M, c, l, indices.data_ptr<int>(), weight.data_ptr<float>(), feat.data_ptr<float>(), out.data_ptr<float>());
  return out;
}
    
//top_grad: (N, c), indices: (N, 3), batch_index: (N, ) -> bottom_grad: (b,c,s), s=r^3
at::Tensor deterministic_devoxelize_backward(
    const at::Tensor top_grad, // [M, c]
    const at::Tensor indices,  // [M, 8]
    const at::Tensor weight,   // [M, 8]
    int N                      // N
){
  int M = top_grad.size(0);
  int c = top_grad.size(1);
  int l = weight.size(1);

  at::Tensor bottom_grad_int = torch::zeros({N, c}, at::device(top_grad.device()).dtype(at::ScalarType::Int));
  deterministic_devoxelize_grad_wrapper(M, N, c, l, indices.data_ptr<int>(), weight.data_ptr<float>(), top_grad.data_ptr<float>(), bottom_grad_int.data_ptr<int>());
  
  at::Tensor bottom_grad = bottom_grad_int.to(at::ScalarType::Double);
  bottom_grad /= 1e10; // a hack
  return bottom_grad.to(at::ScalarType::Float);
}


/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deterministic_devoxelize_forward", &deterministic_devoxelize_forward, "Devoxelization forward (CUDA)");
  m.def("deterministic_devoxelize_backward", &deterministic_devoxelize_backward, "Devoxelization backward (CUDA)");
}
*/
