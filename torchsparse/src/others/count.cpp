#include <torch/torch.h>
#include <vector>
#include "count_gpu.h"


/*
idx: [N]
s: output size
*/
at::Tensor count_forward(
    const at::Tensor idx,
    const int s
)
{
  //return group_point_forward_gpu(points, indices);
  int N = idx.size(0);
  at::Tensor out = torch::zeros({s}, at::device(idx.device()).dtype(at::ScalarType::Int));
  count_wrapper(N, idx.data_ptr<int>(), out.data_ptr<int>());
  return out;
}


/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("count_forward", &count_forward, "Counting forward (CUDA)");
}
*/

