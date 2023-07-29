"""
main
"""
from __future__ import annotations

import asyncio

from src.modules.lower_layer_modules.Exceptions import Error
from src.modules.steps.step07 import step07


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
        # step06()
        step07()

    except KeyboardInterrupt:
        exit(1)
    except Error as error:
        print(error.args[0])
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
