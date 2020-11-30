from absl import app

import kblocks.configs  # pylint: disable=unused-import
import pcn.configs  # pylint: disable=unused-import
from kblocks import cli


def cli_main(argv):
    cli.summary_main(cli.get_gin_summary(argv))


if __name__ == "__main__":
    app.run(cli_main)
