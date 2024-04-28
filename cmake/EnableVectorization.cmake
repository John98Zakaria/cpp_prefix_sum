function(
        enable_vectorization
        project_name
        with_report
)
    if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(VECTORIZATION_FLAGS
                "-ffast-math"
        )
        set(VECTORIZATION_REPORT
                "-Rpass=loop-vectorize"
                "-Rpass-missed=loop-vectorize"
                "-Rpass-analysis=loop-vectorize"
                "-fsave-optimization-record"
        )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(VECTORIZATION_FLAGS
                "-ftree-vectorize"
                "-ffast-math"
        )
        set(VECTORIZATION_REPORT
                "-fopt-info-vec-missed"
        )
    else ()
        message(FATAL_ERROR "Enable Vectorization was only built for clang and gcc")
    endif ()

    if (with_report)
        list(APPEND VECTORIZATION_FLAGS ${VECTORIZATION_REPORT})
    endif ()

    target_compile_options(
            ${project_name}
            INTERFACE # C++ warnings
            $<$<COMPILE_LANGUAGE:CXX>:${VECTORIZATION_FLAGS}>
            # C warnings
            $<$<COMPILE_LANGUAGE:C>:${VECTORIZATION_FLAGS}>
    )
endfunction()
