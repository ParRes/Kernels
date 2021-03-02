#!/usr/bin/env python

import sys
import fileinput
import string
import os

def codegen(src,pattern,stencil_size,radius,model,dim,usm):
    src.write('// declare the kernel name used in SYCL parallel_for\n')
    if (usm):
        kernel_name = pattern+str(radius)+'_usm'
    else:
        kernel_name = pattern+str(radius)+'_'+str(dim)+'d'
    src.write('template <typename T> class '+kernel_name+';\n\n')
    src.write('template <typename T>\n')
    src.write('void '+pattern+str(radius)+'(sycl::queue & q, const size_t n, ')
    if (usm):
        src.write('const T * in, ')
        src.write('T * out)\n')
    elif (dim==2):
        src.write('sycl::buffer<T, 2> & d_in, ')
        src.write('sycl::buffer<T, 2> & d_out)\n')
    else:
        src.write('sycl::buffer<T> & d_in, ')
        src.write('sycl::buffer<T> & d_out)\n')
    src.write('{\n')
    src.write('  q.submit([&](sycl::handler& h) {\n')
    if (not usm):
        src.write('    auto in  = d_in.template get_access<sycl::access::mode::read>(h);\n')
        src.write('    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);\n')
    if (dim==2):
        for r in range(1,radius+1):
            src.write('    sycl::id<2> dx'+str(r)+'(sycl::range<2> {'+str(r)+',0});\n')
            src.write('    sycl::id<2> dy'+str(r)+'(sycl::range<2> {0,'+str(r)+'});\n')
    src.write('    h.parallel_for<class '+kernel_name+'<T>>(')
    src.write('sycl::range<2> {n-'+str(2*radius)+',n-'+str(2*radius)+'}, ')
    src.write('sycl::id<2> {'+str(radius)+','+str(radius)+'}, ')
    src.write('[=] (sycl::item<2> it) {\n')
    if (dim==2):
        src.write('        sycl::id<2> xy = it.get_id();\n')
        src.write('        out[xy] += ')
    else:
        # 1D indexing the slow way
        #src.write('        auto i = it[0];\n')
        #src.write('        auto j = it[1];\n')
        #src.write('        out[i*n+j] += ')
        # 1D indexing the fast way
        src.write('        const auto i = it[0];\n')
        src.write('        const auto j = it[1];\n')
        src.write('        out[i*n+j] += ')
    if pattern == 'star':
        for i in range(1,radius+1):
            if (dim==2):
                if i > 1:
                    src.write('\n')
                    src.write(19*' ')
                src.write('+in[xy+dx'+str(i)+'] * static_cast<T>('+str(+1./(2.*i*radius))+')')
                src.write('\n'+19*' ')
                src.write('+in[xy-dx'+str(i)+'] * static_cast<T>('+str(-1./(2.*i*radius))+')')
                src.write('\n'+19*' ')
                src.write('+in[xy+dy'+str(i)+'] * static_cast<T>('+str(+1./(2.*i*radius))+')')
                src.write('\n'+19*' ')
                src.write('+in[xy-dy'+str(i)+'] * static_cast<T>('+str(-1./(2.*i*radius))+')')
            else:
                # 1D indexing the slow way
                #if i > 1:
                #    src.write('\n')
                #    src.write(22*' ')
                #src.write('+in[i*n+(j+'+str(i)+')] * static_cast<T>('+str(+1./(2.*i*radius))+')')
                #src.write('\n'+22*' ')
                #src.write('+in[i*n+(j-'+str(i)+')] * static_cast<T>('+str(-1./(2.*i*radius))+')')
                #src.write('\n'+22*' ')
                #src.write('+in[(i+'+str(i)+')*n+j] * static_cast<T>('+str(+1./(2.*i*radius))+')')
                #src.write('\n'+22*' ')
                #src.write('+in[(i-'+str(i)+')*n+j] * static_cast<T>('+str(-1./(2.*i*radius))+')')
                # 1D indexing the fast way
                if i > 1:
                    src.write('\n')
                    src.write(30*' ')
                src.write('+in[i*n+(j+'+str(i)+')] * static_cast<T>('+str(+1./(2.*i*radius))+')')
                src.write('\n'+30*' ')
                src.write('+in[i*n+(j-'+str(i)+')] * static_cast<T>('+str(-1./(2.*i*radius))+')')
                src.write('\n'+30*' ')
                src.write('+in[(i+'+str(i)+')*n+j] * static_cast<T>('+str(+1./(2.*i*radius))+')')
                src.write('\n'+30*' ')
                src.write('+in[(i-'+str(i)+')*n+j] * static_cast<T>('+str(-1./(2.*i*radius))+')')
            if i == radius:
                src.write(';\n')
    else:
        print('grid not implemented\n')
    src.write('    });\n')
    src.write('  });\n')
    src.write('}\n\n')

def instance(src,model,pattern,r):
    if pattern == 'star':
        stencil_size = 4*r+1
    else:
        stencil_size = (2*r+1)**2
    codegen(src,pattern,stencil_size,r,model,1,False)
    codegen(src,pattern,stencil_size,r,model,2,False)
    codegen(src,pattern,stencil_size,r,model,1,True)

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

