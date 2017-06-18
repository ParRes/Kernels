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

    std::string loadProgram(std::string input)
    {
        std::ifstream stream(input.c_str());
        if (!stream.is_open()) {
            //std::cout << "Cannot open file: " << input << std::endl;
            return std::string("FAIL");
        }

        return std::string( std::istreambuf_iterator<char>(stream),
                            std::istreambuf_iterator<char>() );
    }

    void OpenCLinfo()
    {
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        if ( all_platforms.size() == 0 ) {
            std::cout<<" No platforms found. Check OpenCL installation!\n";
            exit(1);
        }
        std::cout << "The first of the following is the default and will be used.\n" << "\n";
        for (auto i : all_platforms) {
            std::cout << "Available OpenCL platform: " << i.getInfo<CL_PLATFORM_NAME>() << "\n";
        }
    }
}
#endif // PRK_OPENCL_HPP
