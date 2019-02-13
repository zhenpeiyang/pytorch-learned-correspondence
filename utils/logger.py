import logging
import shutil
import sys
import time

logger = logging.getLogger()


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
  attr = []
  num = color2num[color]
  if highlight:
      num += 10
  attr.append(str(num))
  if bold:
      attr.append('1')
  return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def colored_log(prompt, texts, color='green', bold=True, highlight=False):
  """Show colored logs.
  """
  assert isinstance(prompt, str)
  assert isinstance(texts, str)
  assert isinstance(color, str)
  colored_prompt = colorize(prompt, color, bold=bold, highlight=highlight)
  clean_line = ''
  sys.stdout.write(clean_line)
  logger.info(colored_prompt + texts)


def callback_log(texts):
  """Callback_log will show caller's location.

  Args:
      texts (str): Text to show.

  """
  colored_log('Trigger callback: ', texts)


def warning_log(texts):
  """Warning_log will show caller's location and red texts.

  Args:
      texts (str): Text to show.

  """
  colored_log('Warning: ', texts, color='red')


def error_log(texts):
  """Error_log will show caller's location, red texts and raise
  RuntimeError.

  Args:
      texts (str): Text to show.

  """
  colored_log('Error: ', texts, color='red')
  raise RuntimeError