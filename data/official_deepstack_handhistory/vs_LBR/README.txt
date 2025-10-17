This directory contains logs of the local best response (LBR) playing against agents created by University of Alberta and DeepStack. We do not have a permission to publish these logs for other evaluated ACPC agents.

The files contain hand histories of the matches. The final number in the brackets indicates the variance reduced outcome of the first player, computed based on imaginary observations. The first lines on the files before the hand histories specify the LBR setting, i.e., the actions evaluated by LBR on each street.

The directory indicates the evaluated agent and the file names include the parameters of the evaluation. In all file names the sSEED or rSEED indicate the SEED used for generating cards where the "r" or "s" indicate the side of the cards played by the player. The second part of the file name (separated by _) always indicates the betting options evaluated by LBR.

In DeepStack logs, _ds_ indicates the default setting of DeepStack betting options. dsFCPA is DeepStack with betting restricted only to fold, call, pot bet, and all in. dsMORE is DeepStack with additional betting options, as described on the supplement of the paper.

