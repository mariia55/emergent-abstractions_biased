# Written by Eosandra Grund (egrund@uni-osanbrueck.de)

# Originally from egg.core (file: gs_wrappers.py): https://github.com/facebookresearch/EGG/tree/main/egg/core
# adapted with the methods of: https://github.com/MathieuRita/Lazimpa

from typing import Optional
import torch
import torch.nn as nn
from egg.core import LoggingStrategy

class LazImpaSenderReceiverRnnGS(nn.Module):
    """
    This class implements the Sender/Receiver game mechanics for the Sender/Receiver game with variable-length
    communication messages and Gumber-Softmax relaxation of the channel. The vocabulary term with id `0` is assumed
    to the end-of-sequence symbol. It is assumed that communication is stopped either after all the message is processed
    or when the end-of-sequence symbol is met.

    Laziness: 'length_cost * step * accuracy**threshold' added to the step-loss (therefore a little bit different than Rita et al. )
    Impatience: Already Impatient

    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(10, 5)
    ...     def forward(self, x, _input=None, aux_input=None):
    ...         return self.fc(x)
    >>> sender = Sender()
    >>> sender = RnnSenderGS(sender, vocab_size=2, embed_dim=3, hidden_size=5, max_len=3, temperature=5.0, cell='gru')
    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(7, 10)
    ...     def forward(self, x, _input=None, aux_input=None):
    ...         return self.fc(x)
    >>> receiver = RnnReceiverGS(Receiver(), vocab_size=2, embed_dim=4, hidden_size=7, cell='rnn')
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, labels, aux_input):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {'aux': torch.zeros(sender_input.size(0))}
    >>> game = SenderReceiverRnnGS(sender, receiver, loss)
    >>> loss, interaction = game(torch.ones((3, 10)), None, None)  # batch of 3 10d vectors
    >>> interaction.aux['aux'].detach()
    tensor([0., 0., 0.])
    >>> loss.item() > 0
    True
    """
    
    def __init__(
        self,
        sender,
        receiver,
        loss,
        length_cost=0.0,
        threshold=1.0,
        impatience=True,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset. The auxiliary information should contain acc
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param threshold: the threshold from which the penalty for symbols becomes important
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in the callbacks.

        """
        super(LazImpaSenderReceiverRnnGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost
        self.threshold = threshold
        self.impatience = impatience
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss = 0
        pressure = 0
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0

        aux_info = {}
        z = 0.0
        for step in range(receiver_output.size(1)):
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
                aux_input,
            )
            eos_mask = message[:, step, 0]  # always eos == 0

            # for laziness we need this: 
            #**************************
            #intensity of Rita et al. = 10 == length_cost = 0.1; threshold = beta1 = 45
            step_length_pressure = self.length_cost * step_aux['acc']**self.threshold * (1.0 + step)#**((1.0 + step - 1.75)*0.5) # means eos is assumed as a symbol, therefore message (3,0,0,0) is calcuated like message length 2 !!!

            add_mask = eos_mask * not_eosed_before
            
            if self.impatience:
                z += not_eosed_before
                loss += step_loss * not_eosed_before
            else:
                z += add_mask
                loss += step_loss * add_mask
            pressure += step_length_pressure * add_mask # Laziness: added accuracy (laziness)
            expected_length += add_mask.detach() * (1.0 + step)

            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += step_loss * not_eosed_before
        pressure += step_length_pressure * not_eosed_before
        expected_length += (step + 1) * not_eosed_before

        z += not_eosed_before

        # get average loss, or nothing if not impatience
        if self.impatience:
            loss /= z # get average (z is number of used step_losses)

        # add length pressure
        loss += pressure

        if not self.impatience:
            assert z.allclose(
                torch.ones_like(z)
            ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)

        aux_info["o_loss"] = loss - pressure
        aux_info["pressure"] = pressure
        aux_info["length"] = expected_length

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=expected_length.detach(),
            aux=aux_info,
        )

        return loss.mean(), interaction