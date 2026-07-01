# Error / difference patterns: EMB vs SFL+RFF

==============================================================================
TUCKUTE (control fMRI, lang-LH BOLD)  (n=2000, unit=sentence)
==============================================================================
predictivity: EMB_sum r=0.438  SFL+RFF r=0.354  agreement Spearman=0.532
corr(disagreement d, sentence length (words)) = +0.025   [EMB-favored wc=6.0 ... ]
  mean sentence length (words):  EMB-favored=6.01   SFL+RFF-favored=6.01

  rated-sentence subset n=1000. Group means (EMB-favored vs SFL+RFF-favored):
  feature                     EMB-fav  RFF-fav  Δ(EMB-RFF)
  rating_imageability_mean       3.52     4.15       -0.63  <-- semantic
  rating_physical_mean           2.48     3.56       -1.08  <-- semantic
  rating_places_mean             1.84     2.63       -0.79  <-- semantic
  rating_others_thoughts_mean     4.10     3.52        0.59  <-- semantic
  rating_valence_mean            3.16     3.28       -0.11  <-- semantic
  rating_arousal_mean            3.41     3.04        0.38  <-- semantic
  rating_sense_mean              6.64     6.31        0.33  <-- semantic
  avg_word_length                4.19     4.91       -0.72
  avg_word_freq_zipf             5.86     5.40        0.45
  rating_gram_mean               6.46     6.16        0.30
  dep_length_mean                1.78     1.83       -0.06

  EMB-favored (EMB predicts high, SFL+RFF low):
    d=+4.17 y=-0.07  "And -- and -- and founder of ImClone."
    d=+3.91 y=+0.18  "Ships from and sold by GETPLUS."
    d=+3.70 y=+1.14  "Yes, I am a induced Sep."
    d=+3.39 y=-0.41  "One, two, three, four, five, six-"
    d=+3.14 y=+0.08  "If the Downloading will not start."
    d=+2.99 y=+0.88  "Toomey is up by double Fr."
    d=+2.88 y=+0.08  "Did permission have to be granted?"
    d=+2.80 y=+0.81  "This entry was posted in A.C.T."

  SFL+RFF-favored (SFL+RFF predicts high, EMB low):
    d=-3.78 y=-0.48  "Australias Dillon authoritarian shopping department Nab."
    d=-3.35 y=+0.21  "Sixteen settled shoes around the courtyard."
    d=-3.24 y=+0.18  "Spread over 1.65 million-square-feet Warner Bros."
    d=-3.04 y=+0.14  "When are out in exhibit already."
    d=-3.00 y=-0.35  "Alix and Sway shouted in prayers."
    d=-2.93 y=+0.28  "Long chandelier avocado colored earrings glistened."
    d=-2.91 y=-0.85  "Armor mir circled to drink coffee."
    d=-2.89 y=+0.94  "Wonder then... let's get flying started."

==============================================================================
PEREIRA (fMRI EffectSize)  (n=627, unit=sentence)
==============================================================================
predictivity: EMB_sum r=0.429  SFL+RFF r=0.476  agreement Spearman=0.561
corr(disagreement d, sentence length (words)) = -0.219   [EMB-favored wc=11.7 ... ]
  mean sentence length (words):  EMB-favored=11.68   SFL+RFF-favored=13.55

  EMB-favored (EMB predicts high, SFL+RFF low):
    d=+3.25 y=+1.31  "Every night while we sleep, our minds are as active as when we are"
    d=+2.56 y=+0.64  "The fovea is the center of the human eye and the most sensitive pa"
    d=+2.46 y=+0.72  "Good data on the social and economic effects of legalized gambling"
    d=+2.41 y=+1.01  "Murder is the unlawful killing of another human being with intent."
    d=+2.12 y=+0.21  "Freshly squeezed orange juice often contains pulp as well."
    d=+2.08 y=+0.94  "Assault rifles can fire in bursts and are the standard infantry we"
    d=+2.05 y=+0.26  "Dairy products such as cheese and butter are made from milk."
    d=+1.94 y=+0.51  "Private law school tuition has increased by a factor of four since"

  SFL+RFF-favored (SFL+RFF predicts high, EMB low):
    d=-3.29 y=+0.45  "A banana is a long fruit that grows in bunchs, with a soft edible "
    d=-2.86 y=+0.84  "Forks are used in Western cultures, and there are special kinds fo"
    d=-2.40 y=+1.41  "The piano repertoire is large and famous pianists can give solo co"
    d=-2.39 y=+0.66  "Rain is water that falls from clouds in the sky, which are condens"
    d=-2.38 y=+0.73  "A desk is a piece of furniture used for working at, sometimes with"
    d=-2.23 y=+1.14  "Some patients go there for specialist diagnosis or therapy, leavin"
    d=-2.20 y=+0.07  "A raven is a large black bird that thrives in many climates."
    d=-2.06 y=+0.56  "Some refrigerators can have a small compartment at freezing temper"

==============================================================================
UCL-N400 (ERP amplitude)  (n=1505, unit=word)
==============================================================================
predictivity: EMB_sum r=0.298  SFL+RFF r=0.349  agreement Spearman=0.441
corr(disagreement d, word length (chars)) = +0.081   [EMB-favored len=3.9 ... ]
  mean word length (chars):  EMB-favored=4.23   SFL+RFF-favored=3.92

  EMB-favored (EMB predicts high, SFL+RFF low):
    d=+3.47 y=+1.36  "I'll"  | You give me a lead and I'll follow it.
    d=+3.26 y=-0.79  "just"  | She threatened to leave before and this time she just had enough
    d=+3.23 y=-0.81  "till"  | Billy knocked on the door and waited till he was told to enter.
    d=+3.14 y=+0.43  "mostly"  | I mostly just pick a dry white and drink it to get drunk.
    d=+3.09 y=-0.47  "uses"  | He uses his belt on you.
    d=+3.09 y=-3.16  "Lucy"  | Lewis sat down and Lucy laid her hand on his cheek.
    d=+2.99 y=+1.34  "so"  | He knocked on the door so hard that he almost broke it down.
    d=+2.93 y=+0.32  "smiled"  | He smiled again and felt like a man and not just the kid.

  SFL+RFF-favored (SFL+RFF predicts high, EMB low):
    d=-5.30 y=-1.06  "single"  | They say he can breathe fire and kill a man with a single word.
    d=-4.75 y=-1.88  "an"  | James was only in the cell for an hour but it felt like a week.
    d=-3.60 y=+0.36  "good"  | If this were a movie instead of a book this would be a good bit.
    d=-3.51 y=-1.45  "off"  | The shout caught him off guard and he fell back a little way.
    d=-3.36 y=-0.29  "last"  | All last week there was a white van parked across the street.
    d=-3.36 y=-0.86  "a"  | They say he can breathe fire and kill a man with a single word.
    d=-3.20 y=-0.38  "first"  | You shouldn't have been racing my truck in the first place.
    d=-3.07 y=-0.69  "straight"  | In the morning Mickey stood straight with his hand behind his ba
