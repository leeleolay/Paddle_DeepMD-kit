# libmd

set (libname ${LIB_DEEPMD_CC})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

if (USE_CUDA_TOOLKIT)
  include_directories("${CUDA_INCLUDE_DIRS}")
endif()

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

add_library(${libname} SHARED ${LIB_SRC})

if (USE_CUDA_TOOLKIT)
  target_link_libraries (${libname} ${CUDA_LIBRARIES})
endif()

if(PADDLE_FOUND)
  target_link_libraries(${libname}
      ${PADDLE_LIBRARIES}
      ${PADDLE_THIRD_PARTY_LIBRARIES}
      rt dl pthread)
  if (mklml_FOUND)
    target_link_libraries(${libname} "-L${THIRD_PARTY_ROOT}/install/mklml/lib -liomp5 -Wl,--as-needed")
  endif()

else()
  message(FATAL_ERROR "Cannot find PaddlePaddle Fluid under ${PADDLE_ROOT}")
endif()


install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)