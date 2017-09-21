"""
General logging class for PyTorch training loops.
"""
import os
import torch

class Logger(object):
    """
    Abstracted class to expose logging methods to a series of files on the run directory.
    """
    def __init__(self, run_dir):
        """
        Construct logger and keep files open.
        """
        ### specify directories; create directories if they don't exist:
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.ckpt_dir = os.path.join(run_dir, 'ckpts/')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.log_dir = os.path.join(run_dir, 'logs/')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        ### create output logging file and keep open:
        self.training_log = os.path.join(self.log_dir, 'training.log')
        self._training_log_f = open(self.training_log, 'w')
        self.messages_log = os.path.join(self.log_dir, 'messages.log')
        self._messages_log_f = open(self.messages_log, 'w')


    def close(self):
        """
        Close all file handles.
        """
        self._training_log_f.close()
        self._messages_log_f.close()


    def log(self, loss, step, train, valid):
        """
        Log a loss message to the logfile.
        """
        self._training_log_f.write(
            "{0} @ step: {1} | Training: {2:.4f} | Validation {3:.4f} \n".format(loss, step, train, valid))
        self._training_log_f.flush()


    def save(self, timestep, model_core, model_ctc):
        """Save model to run directory."""
        _core_model_path = os.path.join(self.ckpt_dir, "wavenet_core.t{}.pt".format(timestep))
        _ctc_model_path = os.path.join(self.ckpt_dir, "wavenet_ctc.t{}.pt".format(timestep))
        torch.save(model_core.state_dict(), _core_model_path)
        torch.save(model_ctc.state_dict(), _ctc_model_path)
        self._messages_log_f.write(
            "[{0}] Saved wavenet base models to: {1}, {2}\n".format(timestep, _core_model_path, _ctc_model_path))
        self._messages_log_f.flush()
