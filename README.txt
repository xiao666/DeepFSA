The useful files are:

cal_news.py: calculate the financial sentimen scores for given news using certain model

SemEval_Full: fine tune with Full
SemEval_Chain_Thaw: fine tuen with chain thaw
SemEval_New: fine tuen with new
SemEval_Last: fine tuen with last

sp_lstm: a lstm model using single/concatenated factors for stock prediction

---------------------------------------------------------------------------------------------

Some csv files mean:
daily_news: Reddits top 25 daily news, 25 news per day * 1989 days
Headlines_Testdata_withscores: test set of Track 2 of SemEval2017 Task5
Headlines_Trainingdata: train set of Track 2
Microblog_Trainingdata: train set of Track 1
Microblogss_Testdata_withscores: test set of Track 1

For using test sets of two Tracks, fill a form at:
http://alt.qcri.org/semeval2017/task5/index.php?id=data-and-tools

returns: previous daily return
ndr: normalized daily return of previous day
scores1: sentiments scores of Reddits news using DeepFSA_Full_Mciroblogs
scores2: sentiments scores of Reddits news using DeepFSA_Full_News



---------------------------------------------------------------------------------------------


Saved models to calculate the sentiment scores of new News or Microblogs messages are available at:
https://drive.google.com/drive/folders/12LnYpIUK9_6-2nIY4Bm3zeW6fExzMNAi?usp=sharing


----------------------------------------------------------------------------------------------


