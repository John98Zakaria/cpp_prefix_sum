function(
        enable_intrinsics
        project_name
)

    if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(SIMD_FLAGS "-ffast-math" "-march=native")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(SIMD_FLAGS "-ffast-math" "-march=native")
    else ()
        message(FATAL_ERROR "Enable Intrinsics was only built for clang and gcc")
    endif ()

    target_compile_options(
            ${project_name}
            INTERFACE # C++ warnings
            $<$<COMPILE_LANGUAGE:CXX>:${SIMD_FLAGS}>
            # C warnings
            $<$<COMPILE_LANGUAGE:C>:${SIMD_FLAGS}>
    )
endfunction()
