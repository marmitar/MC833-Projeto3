import os
import sys
from argparse import Action, ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Sequence
from enum import Enum
from typing import Any, Final, Literal, LiteralString, final, get_args, override

import colored_traceback

__all__ = ['ColorOutputAction', 'add_color_option']


def _enable_colors(default: bool | None = None) -> bool:
    """
    Checks if the environment supports colors and the user wants them.

    ## References
    - <https://bixense.com/clicolors/>
    - <https://no-color.org/>
    """
    if os.environ.get('NO_COLOR'):
        return False
    elif os.environ.get('CLICOLOR_FORCE'):
        return True
    elif default is None or os.environ.get('CLICOLOR'):
        return sys.stdout.isatty()
    else:
        return default


@final
class AutoColorOutput(Enum):
    """
    Automatically checks if the environment supports colors and the user wants them.

    ## References
    - <https://bixense.com/clicolors/>
    - <https://no-color.org/>
    """

    AUTO = ()

    @property
    def enabled(self) -> bool:
        """
        Checks if the environment supports colors and the user wants them.
        """
        return _enable_colors()

    def __bool__(self) -> bool:
        return _enable_colors()

    @override
    def __str__(self) -> Literal['auto']:
        return 'auto'


# Valid options for '--color' according to <https://bixense.com/clicolors/>.
type ColorOutputOption = Literal['', 'always', 'on', 'never', 'no', 'auto']


def _color_output_option(option: ColorOutputOption | str | None) -> bool | AutoColorOutput:
    """
    Translate string option into color option, following <https://bixense.com/clicolors/>.
    """
    match option:
        case '' | 'always' | 'on':
            return True
        case 'never' | 'no':
            return False
        case 'auto':
            return AutoColorOutput.AUTO
        case _:
            raise ArgumentTypeError(f'invalid color option: {option}')


def _set_colorer_traceback(enable: bool | AutoColorOutput) -> None:
    """
    Add or remove hook for colored taceback.
    """
    if enable:
        colored_traceback.add_hook(always=True)
    else:
        sys.excepthook = sys.__excepthook__


@final
class ColorOutputAction(Action):
    """
    Parse `--color` and `--no-color` options from command line.
    """

    OPTION_STRINGS: Final = ('--color', '--no-color')

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = 'color',
        default: ColorOutputOption | AutoColorOutput | bool | None = AutoColorOutput.AUTO,
        required: bool = False,
        help: str | None = None,
    ) -> None:
        if set(option_strings) != set(self.OPTION_STRINGS):
            raise ValueError(f'invalid color options: {", ".join(option_strings)}')

        if isinstance(default, str):
            default = _color_output_option(default)
        elif default is None:
            default = AutoColorOutput.AUTO

        super().__init__(
            option_strings=self.OPTION_STRINGS,
            dest=dest,
            nargs='?',
            default=default,
            type=_color_output_option,
            choices=get_args(ColorOutputOption),
            required=required,
            help=help,
        )

        _set_colorer_traceback(default)

    @override
    def __call__(
        self,
        _parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ):
        match option_string:
            case '--color':
                if isinstance(values, str):
                    enable_colors = _color_output_option(values)
                elif values is None:
                    enable_colors = True
                elif len(values) == 1:
                    enable_colors = _color_output_option(values[0])
                else:
                    raise ArgumentTypeError(f'multiple color options: {", ".join(values)}')

            case '--no-color':
                if values is None:
                    enable_colors = False
                else:
                    raise ArgumentTypeError(f'invalid --no-color argument: {values}')

            case _:
                return

        setattr(namespace, self.dest, enable_colors)
        _set_colorer_traceback(enable_colors)

    @override
    def format_usage(self) -> LiteralString:
        return ' | '.join(self.OPTION_STRINGS)


def add_color_option(
    parser: ArgumentParser,
    *,
    default: ColorOutputOption | AutoColorOutput | bool | None = AutoColorOutput.AUTO,
    help: str = 'Display colored output.',
    dest: str = 'color',
) -> Action:
    """
    Add `--color` and `--no-color` options to the command line.
    """
    return parser.add_argument(
        *ColorOutputAction.OPTION_STRINGS,
        action=ColorOutputAction,
        default=default,
        help=help,
        dest=dest,
    )
