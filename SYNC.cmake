if(NOT DEFINED REPO_ROOT)
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

set(METASDT_BACKEND_SRC_DIR "${REPO_ROOT}/Cpp/src")
set(METASDT_BACKEND_INCLUDE_DIR "${REPO_ROOT}/Cpp/include/metaSDT")

set(METASDT_R_SRC_DEST "${REPO_ROOT}/R/src/cpp")
set(METASDT_R_INCLUDE_DEST "${REPO_ROOT}/R/inst/include/metaSDT")
set(METASDT_PY_SRC_DEST "${REPO_ROOT}/Python/src/cpp")
set(METASDT_PY_INCLUDE_DEST "${REPO_ROOT}/Python/src/include/metaSDT")

foreach(sync_dir
        "${METASDT_R_SRC_DEST}"
        "${METASDT_R_INCLUDE_DEST}"
        "${METASDT_PY_SRC_DEST}"
        "${METASDT_PY_INCLUDE_DEST}")
    file(REMOVE_RECURSE "${sync_dir}")
    file(MAKE_DIRECTORY "${sync_dir}")
endforeach()

file(GLOB METASDT_BACKEND_SOURCES "${METASDT_BACKEND_SRC_DIR}/*.cpp")
file(GLOB METASDT_BACKEND_HEADERS "${METASDT_BACKEND_INCLUDE_DIR}/*.hpp")

foreach(source_file ${METASDT_BACKEND_SOURCES})
    file(COPY "${source_file}" DESTINATION "${METASDT_R_SRC_DEST}")
    file(COPY "${source_file}" DESTINATION "${METASDT_PY_SRC_DEST}")
endforeach()

foreach(header_file ${METASDT_BACKEND_HEADERS})
    file(COPY "${header_file}" DESTINATION "${METASDT_R_INCLUDE_DEST}")
    file(COPY "${header_file}" DESTINATION "${METASDT_PY_INCLUDE_DEST}")
endforeach()

set(METASDT_SYNC_NOTE
"Do not edit files in this directory directly.
Edit root Cpp/src and Cpp/include/metaSDT, then run CMake to synchronize.
")

file(WRITE "${METASDT_R_SRC_DEST}/README.md" "${METASDT_SYNC_NOTE}")
file(WRITE "${METASDT_R_INCLUDE_DEST}/README.md" "${METASDT_SYNC_NOTE}")
file(WRITE "${METASDT_PY_SRC_DEST}/README.md" "${METASDT_SYNC_NOTE}")
file(WRITE "${METASDT_PY_INCLUDE_DEST}/README.md" "${METASDT_SYNC_NOTE}")

message(STATUS "metaSDT: synchronized backend C++ sources and headers.")
