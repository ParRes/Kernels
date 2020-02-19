// Taken from https://github.com/HandsOnOpenCL/Exercises-Solutions/blob/master/Exercises/Cpp_common/util.hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// This work is licensed under the Creative Commons Attribution 3.0 Unported License.
//
// To view a copy of this license, visit http://creativecommons.org/licenses/by/3.0/
// or send a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef PRK_OPENCL_HPP
#define PRK_OPENCL_HPP

#include <iostream>
#include <fstream>
#include <string>

#include <cstdlib>

#include "cl.hpp"

namespace prk {

  bool stringContains(std::string const & a, std::string const & b)
  {
      std::string::size_type n = a.find(b);
      return (n != std::string::npos);
  }

  namespace opencl {

    std::string loadProgram(std::string input)
    {
      std::ifstream stream(input.c_str());
      if (!stream.is_open()) {
        return std::string("FAIL");
      }

      return std::string( std::istreambuf_iterator<char>(stream),
                          std::istreambuf_iterator<char>() );
    }

    bool listPlatforms()
    {
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);
      if ( platforms.size() == 0 ) {
        std::cout <<" No platforms found. Check OpenCL installation!\n";
        return false;
      }
      std::cout << "Available OpenCL platforms: \n";
      auto first = true;
      for (auto i : platforms) {
        std::cout << "CL_PLATFORM_NAME=" << i.getInfo<CL_PLATFORM_NAME>() << ", ";
        std::cout << "CL_PLATFORM_VENDOR=" << i.getInfo<CL_PLATFORM_VENDOR>();
        if ( first ) { std::cout << " (DEFAULT)"; first = false; }
        std::cout << std::endl;

        std::vector<cl::Device> devices;
        i.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto j : devices) {
          std::cout << "   CL_DEVICE_NAME="                << j.getInfo<CL_DEVICE_NAME>()      << "\n";
          std::cout << "   CL_DEVICE_VENDOR="              << j.getInfo<CL_DEVICE_VENDOR>()    << "\n";
          std::cout << "   CL_DEVICE_AVAILABLE="           << j.getInfo<CL_DEVICE_AVAILABLE>() << "\n";
          auto t = j.getInfo<CL_DEVICE_TYPE>();
          std::string s;
          switch (t) {
              case CL_DEVICE_TYPE_CPU:         s="CPU"; break;
              case CL_DEVICE_TYPE_GPU:         s="GPU"; break;
              case CL_DEVICE_TYPE_ACCELERATOR: s="ACCELERATOR"; break;
              //case CL_DEVICE_TYPE_CUSTOM:      s="CUSTOM"; break;
              default: s="UNKNOWN"; break;
          }
          std::cout << "   CL_DEVICE_TYPE="                << s << "\n";
          std::cout << "   CL_DEVICE_MAX_COMPUTE_UNITS="   << j.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()   << "\n";
          std::cout << "   CL_DEVICE_GLOBAL_MEM_SIZE="     << j.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()     << "\n";
          std::cout << "   CL_DEVICE_MAX_CLOCK_FREQUENCY=" << j.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";
          std::cout << "   CL_DEVICE_MAX_MEM_ALLOC_SIZE="  << j.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()  << "\n";
          std::cout << "   CL_DEVICE_LOCAL_MEM_SIZE="      << j.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()      << "\n";
          //std::cout << "   CL_DEVICE_EXTENSIONS="          << j.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";
          auto e = j.getInfo<CL_DEVICE_EXTENSIONS>();
          auto has64 = stringContains(e,"cl_khr_fp64");
          std::cout << "   CL_DEVICE_EXTENSIONS " << (has64 ? "contains" : "does not contain") << " cl_khr_fp64\n";
          std::cout << std::endl;
        }
      }
      return true;
    }

    int precision(cl::Context context) {
      bool has64 = true;
      std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
      for (auto j : devices) {
        auto e = j.getInfo<CL_DEVICE_EXTENSIONS>();
        has64 &= stringContains(e,"cl_khr_fp64");
      }
      return ( has64 ? 64 : 32 );
    }

    bool available(cl::Context context) {
      std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
      if ( devices.size() == 0 ) return false;
      bool avail = true;
      for (auto j : devices) {
        auto a = j.getInfo<CL_DEVICE_AVAILABLE>();
        avail &= (a==1);
      }
      return ( avail );
    }
  }
}
#endif // PRK_OPENCL_HPP
