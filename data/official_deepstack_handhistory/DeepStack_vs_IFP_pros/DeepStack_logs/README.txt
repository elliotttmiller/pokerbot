The logs from DeepStack's matches are provided in several formats in the following sub-directories:

ACPC/
PokerStars/DeepStack_view
PokerStars/full_info
PokerStars/participant_view

In each of the sub-directories, there is a log for each individual participant's match along with an aggregate log for the entire study, all of which are sorted chronologically by the time each hand started.

Logs in the ACPC directory use the Annual Computer Poker Competition format and are the study's official logs.  The directory also contains a brief README file that explains the ACPC format.

To make it easier for people within the online poker community to examine the matches, we also provide three additional logs for each match that are consistent with the format used by PokerStars: one showing both players' private cards, one from the participant's view of the game (only revealing DeepStack's cards at showdowns), and one from DeepStack's view.  The PokerStars format is usually supported by existing poker analytics software and you should be able to import the logs.  However, note that the PokerStars logs are synthetic in a sense, as we only mimic the PokerStars format.  In particular, the hand numbers are generated (from 1 to the number of hands the participant played against DeepStack) and may conflict with existing PokerStars hands in tools like Hold'em Manager.  While it is unlikely that you have genuine PokerStars hands with duplicate hand numbers in this range, you may want to use a separate database to avoid any potential issues if you plan to import these logs.
