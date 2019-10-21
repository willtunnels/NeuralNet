find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(neuralnet
  Network.cpp
  main.cpp
  mnist.cpp
)

target_link_libraries(neuralnet
  PRIVATE
    Eigen3::Eigen
)
