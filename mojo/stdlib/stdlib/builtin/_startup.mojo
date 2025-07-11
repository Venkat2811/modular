# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements functionality to start a mojo execution."""

from sys import external_call
from sys.ffi import _get_global


fn _init_global_runtime() -> OpaquePointer:
    return external_call[
        "KGEN_CompilerRT_AsyncRT_CreateRuntime",
        OpaquePointer,
    ](0)


fn _destroy_global_runtime(ptr: OpaquePointer):
    """Destroy the global runtime if ever used."""
    external_call["KGEN_CompilerRT_AsyncRT_DestroyRuntime", NoneType](ptr)


@always_inline
fn _ensure_current_or_global_runtime_init():
    var current_runtime = external_call[
        "KGEN_CompilerRT_AsyncRT_GetCurrentRuntime", OpaquePointer
    ]()
    if current_runtime:
        return
    _ = _get_global["Runtime", _init_global_runtime, _destroy_global_runtime]()


fn __wrap_and_execute_main[
    main_func: fn () -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for non-raising main function."""

    # Initialize the global runtime.
    _ensure_current_or_global_runtime_init()

    # Initialize the mojo argv with those provided.
    external_call["KGEN_CompilerRT_SetArgV", NoneType](argc, argv)

    # Call into the user main function.
    main_func()

    # Delete any globals we have allocated.
    external_call["KGEN_CompilerRT_DestroyGlobals", NoneType]()

    # Return OK.
    return 0


fn __wrap_and_execute_raising_main[
    main_func: fn () raises -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for a raising main function."""

    # Initialize the global runtime.
    _ensure_current_or_global_runtime_init()

    # Initialize the mojo argv with those provided.
    external_call["KGEN_CompilerRT_SetArgV", NoneType](argc, argv)

    # Call into the user main function.
    try:
        main_func()
    except e:
        print("Unhandled exception caught during execution:", e)
        return 1

    # Delete any globals we have allocated.
    external_call["KGEN_CompilerRT_DestroyGlobals", NoneType]()

    # Return OK.
    return 0


# A prototype of the main entry point, used by the compiled when synthesizing
# main.
fn __mojo_main_prototype(
    argc: Int32, argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`]
) -> Int32:
    return 0
