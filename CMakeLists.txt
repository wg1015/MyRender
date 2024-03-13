cmake_minimum_required (VERSION 3.8)

project (VulkanTutorial)
set(ExampleDir "render_example")

message(STATUS "VariableName:${CMAKE_BINARY_DIR}")
message(STATUS "SourceDir is ${CMAKE_SOURCE_DIR}")
message(STATUS "ExampleDir is ${ExampleDir}")

find_package (glfw3 REQUIRED)
find_package (Vulkan REQUIRED)

if(NOT DEFINED Example)
  message(STATUS "Example config is needed.")
endif()

########################################################

message(STATUS "Example name is ${Example}")
add_definitions(-DNDEBUG)
add_executable(${Example} ${CMAKE_SOURCE_DIR}/${ExampleDir}/${Example}/${Example}.cpp)
set_target_properties(${Example} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_target_properties(${Example} PROPERTIES
  CXX_STANDARD 17
)
target_link_libraries (${Example} Vulkan::Vulkan glfw)

########################################################
set (FRAG_SHADER shader.frag)
set (VERT_SHADER shader.vert)
set (SHADERS_DIR shaders)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${SHADERS_DIR}/${Example})

add_custom_command (
  TARGET ${Example}
  PRE_BUILD
  COMMAND glslangValidator -V ${CMAKE_SOURCE_DIR}/${ExampleDir}/${Example}/${FRAG_SHADER} -o ${CMAKE_BINARY_DIR}/${SHADERS_DIR}/${Example}/frag.spv
  COMMAND glslangValidator -V ${CMAKE_SOURCE_DIR}/${ExampleDir}/${Example}/${VERT_SHADER} -o ${CMAKE_BINARY_DIR}/${SHADERS_DIR}/${Example}/vert.spv
  COMMENT "Compiling Shaders"
)













