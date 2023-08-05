"""
main
"""
from __future__ import annotations

import asyncio

from src.modules.lower_layer_modules.Exceptions import Error
from src.modules.steps.step15_16 import step15_16


async def main() -> None:
    """
    main
    """
    # prefix_directory: Path = Path(__file__).parent.parent
    # data_directory: Path = prefix_directory / "data"
    try:
        # step01()
        # step02()
        # step03()
        # step04()
        # step05_06()
        # step07()
        # step08()
        # step09()
        # step11()
        # step12()
        # step13()
        # step14()
        step15_16()

    except KeyboardInterrupt:
        exit(1)
    except (TypeError, Error) as error:
        print(error.args[0])
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
