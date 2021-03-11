__global__ void cuda_GetImgDiff(unsigned char *dest, unsigned char *a, unsigned char *b, int res) {
    int x = 3*threadIdx.x + 3*(blockIdx.x * blockDim.x);
    int y = (3 * res)*threadIdx.y + (3 * res)*(blockIdx.y * blockDim.y);
    int z = threadIdx.z;
    int i = (x + y + z);
    if(a[i] >= b[i]){
        dest[i] = a[i] - b[i];
    }
    else{
        dest[i] = b[i] - a[i];
    }
}

__global__ void cuda_SumPixels(float *d_in, float *d_out) {
        int thId = threadIdx.x;
        int id = threadIdx.x + blockDim.x * blockIdx.x;

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if(thId < s)
            {
                d_in[id] += d_in[id + s];
            }
            __syncthreads();
        }

        if(thId == 0)
        {
            d_out[blockIdx.x] = d_in[id];
        }
}

__global__ void cuda_ByteToFloat(float *f, unsigned char *b) {
      int x = threadIdx.x + blockDim.x * blockIdx.x;
      f[x] = (float)b[x];
}