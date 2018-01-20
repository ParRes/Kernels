#!/usr/bin/env python3

import sys
import fileinput
import string
import os

def codegen(src,pattern,stencil_size,radius,W,model):
    src.write('void '+pattern+str(radius)+'(cl::sycl::queue & q, const size_t n,\n')
    src.write('           cl::sycl::buffer<double, 2> d_in,\n')
    src.write('           cl::sycl::buffer<double, 2> d_out) {\n')
    src.write('  q.submit([&](cl::sycl::handler& h) {\n')
    src.write('    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);       \n')
    src.write('    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);\n')
    src.write('    h.parallel_for<class '+pattern+str(radius)+'>(cl::sycl::range<2> {n-2*'+str(radius)+',n-2*'+str(radius)+'}, cl::sycl::id<2> {'+str(radius)+','+str(radius)+'},\n')
    src.write('                                [=] (cl::sycl::item<2> it) {\n')
    src.write('        cl::sycl::id<2> xy = it.get_id();\n')
    for r in range(1,radius+1):
        src.write('        cl::sycl::id<2> dx'+str(r)+'(cl::sycl::range<2> {'+str(r)+',0});\n')
        src.write('        cl::sycl::id<2> dy'+str(r)+'(cl::sycl::range<2> {0,'+str(r)+'});\n')
    src.write('        out[xy] += ')
    if pattern == 'star':
        for i in range(1,radius+1):
            if i > 1:
                src.write('\n')
                src.write(19*' ')
            src.write('+in[xy+dx'+str(i)+'] * '+str(+1./(2.*i*radius)))
            src.write('\n'+19*' ')
            src.write('+in[xy+dy'+str(i)+'] * '+str(+1./(2.*i*radius)))
            src.write('\n'+19*' ')
            src.write('+in[xy-dx'+str(i)+'] * '+str(-1./(2.*i*radius)))
            src.write('\n'+19*' ')
            src.write('+in[xy-dy'+str(i)+'] * '+str(-1./(2.*i*radius)))
            if i == radius:
                src.write(';\n')
    else:
        print('grid not implemented\n')
    src.write('    });\n')
    src.write('  });\n')
    src.write('}\n\n')

def instance(src,model,pattern,r):

    W = [[0.0e0 for x in range(2*r+1)] for x in range(2*r+1)]
    if pattern == 'star':
        stencil_size = 4*r+1
        for i in range(1,r+1):
            W[r][r+i] = +1./(2*i*r)
            W[r+i][r] = +1./(2*i*r)
            W[r][r-i] = -1./(2*i*r)
            W[r-i][r] = -1./(2*i*r)

    else:
        stencil_size = (2*r+1)**2
        for j in range(1,r+1):
            for i in range(-j+1,j):
                W[r+i][r+j] = +1./(4*j*(2*j-1)*r)
                W[r+i][r-j] = -1./(4*j*(2*j-1)*r)
                W[r+j][r+i] = +1./(4*j*(2*j-1)*r)
                W[r-j][r+i] = -1./(4*j*(2*j-1)*r)

            W[r+j][r+j]    = +1./(4*j*r)
            W[r-j][r-j]    = -1./(4*j*r)

    codegen(src,pattern,stencil_size,r,W,model)

def main():
    for model in ['sycl']:
      src = open('stencil_'+model+'.hpp','w')
      #for pattern in ['star','grid']:
      for pattern in ['star']:
        for r in range(1,6):
          instance(src,model,pattern,r)
      src.close()

if __name__ == '__main__':
    main()

