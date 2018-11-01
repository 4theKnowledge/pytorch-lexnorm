import sys
import time
from colorama import Fore, Back, Style

PROGRESS_BAR_LENGTH = 40

class ProgressBar():
	def __init__(self):
		pass

	def draw_bar(self, i, epoch, num_batches, max_epochs, epoch_start_time):
		epoch_progress = int(PROGRESS_BAR_LENGTH * (1.0 * i / num_batches))
		sys.stdout.write("      Epoch %s of %d : [" % (str(epoch).ljust(len(str(max_epochs))), max_epochs))
		sys.stdout.write("=" * epoch_progress)
		sys.stdout.write(">")
		sys.stdout.write(" " * (PROGRESS_BAR_LENGTH - epoch_progress))
		sys.stdout.write("] %d / %d" % (i, num_batches))
		sys.stdout.write("   %.2fs" % (time.time() - epoch_start_time))
		sys.stdout.write("\r")
		sys.stdout.flush()

	def draw_completed_epoch(self, loss, loss_list, epoch, max_epochs, epoch_start_time, f1_score=None):
		outstr =  ""
		outstr += "      Epoch %s of %d : Loss = %.6f" % (str(epoch).ljust(len(str(max_epochs))), max_epochs, loss)
		if epoch > 1:
			diff = loss - loss_list[epoch - 2]
			col = Fore.GREEN if diff < 0 else "%s+" % Fore.RED
			outstr += " %s%.6f%s" % ( col, loss - loss_list[epoch - 2], Style.RESET_ALL )
		else:
			outstr += " " * 10
		if f1_score is not None:
			outstr += "   F1: %.4f" % f1_score
		else:
			outstr += "             "
		outstr += "   %.2fs" % (time.time() - epoch_start_time)


		sys.stdout.write("%s%s\n" % (outstr, " " * (PROGRESS_BAR_LENGTH - len(outstr) + 60)))
		sys.stdout.flush()