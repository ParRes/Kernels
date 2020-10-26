#include <iostream>
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;
int
main(int argc, char *argv[])
{
  auto platforms = sycl::platform::get_platforms();
    std::cout << platforms.size() << " Platforms" << std::endl;
      for (auto &platform : platforms) {
          std::cout << "Platform: " 
          << platform.get_info<sycl::info::platform::name>()
          << std::endl;
    }

    size_t number = 0;
    for (auto &OnePlatform : platforms) {
      std::cout
	   << ++number << " found..."
	   << std::endl
	   << "Platform: "
	   << OnePlatform.get_info<sycl::info::platform::name>()
	   << std::endl;

      /* Loop through the devices SYCL can find
       * there is always ONE */
      auto MyDevices = OnePlatform.get_devices();
      for (auto &OneDevice : MyDevices ) {
	  std::cout
	   << " Device: "
	   << OneDevice.get_info<sycl::info::device::name>()
	   << std::endl;
      } //devices
      std::cout << std::endl;
     }//platforms
}
