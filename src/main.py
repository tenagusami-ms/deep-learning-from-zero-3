"""
main
"""
from __future__ import annotations

import asyncio

from src.modules.steps.step34 import step34


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
        # step15_16()
        # step17()
        # step18()
        # step19()
        # step20()
        # step21()
        # step22()
        # step23()
        # step24()
        # step25_26()
        # step27()
        # step28()
        # step29()
        # step30_33()
        step34()

    except KeyboardInterrupt:
        exit(1)
    # except (TypeError, Error) as error:
    #     print(error.args[0])
    #     exit(1)


if __name__ == "__main__":
    asyncio.run(main())
