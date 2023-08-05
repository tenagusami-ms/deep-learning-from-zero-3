is_simple_core: bool = True

if is_simple_core:
    from src.modules.dezero.core_simple import Variable
    from src.modules.dezero.core_simple import Function
    from src.modules.dezero.core_simple import using_config
    from src.modules.dezero.core_simple import no_grad
    from src.modules.dezero.core_simple import as_array
    from src.modules.dezero.core_simple import as_variable
