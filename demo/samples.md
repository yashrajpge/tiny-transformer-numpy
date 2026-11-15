# Training demo

One real CPU run of `demo/run_demo.py` (config: dim=64, 4 heads, 2 layers, ctx=48,
batch=16, Adam lr=3e-3, seed=0, 250k-char Tiny Shakespeare, 90/10 split).
Samples are 220 characters drawn with a fixed generator seed at each milestone,
shown verbatim — no cherry-picking.

Batch loss trajectory (1500 steps, ~49 s on a laptop CPU):

| step | loss |
|---|---|
| 1 | 4.1250 |
| 250 | 2.2239 |
| 500 | 2.0082 |
| 750 | 1.9272 |
| 1000 | 1.8733 |
| 1250 | 1.8504 |
| 1500 | 1.8793 |

Final held-out metrics from the same protocol: train loss 1.7203,
val loss 1.8188, val perplexity 6.16.

## Generated text at each stage

### Step 0 (untrained)

```

yMvEI-DIxERbrrgejB;r&ffa&yQV
ErPju.z?w'RpFryqRLS,CUiQIwjSSjqHhns TbFRUz
AEK!JAoc'Ce.x:nSHCtM' QsKnw&HbH?TkeEbHFvGwAYMn.wi?g,W-!NfheZsV,:xm
weu;ZU?&e DVbGpthv&!..Vu?zsIATI ,Zdx
icU.qhYy,iSGsmQcc
CCCiwEj
nm!,WkhfkZE uOiFy?
```

### Step 250 (loss 2.2239)

```

get ae. wemeror. E barioFt, 'dorgr y waniknte:
I ber.
Voootalllm me inv bebal's, blaver, con I vesucen berf, fet wen.

Whas eTd nor. Saeve
Whoe? Har I hatoto I apr yoman Marev: re thew therind 'd coveo tl aprlot, Ilen yo
```

### Step 500 (loss 2.0082)

```

tho han-wears?

QUEN IUHAY NAMANUTUTUS:
Whowk!


DoUSIThe, this:
I fory
Bilan, of set the tha, ulow him jrill fret of,
Your neangenes.

Yo vit Nof pallallf wace nonus con anks son they
me to on eallt tI tearll'ion whimr;
```

### Step 1000 (loss 1.8733)

```

thou be we prousom! I so you
It to you lofitng make, from to thone gokt cheam thank were exhah whould hath,
They we it will realj thoucess with of tail, uthe
AF Romould had swas market they!

MARCHIUS:
Wich this e't. Im 
```

### Step 1500 (loss 1.8793)

```

thus, chold-stoood tengiby o frignaw, Lird; then enso: they,
Too much yand alll all you, as a knows hin nour have?

Filtt.

MAENENIUS:
Cru; stare ar goives; I cantayou me ares speak, yet eto spail, wea till tirme as thy

```

## Tests (`python tests/test_gradients.py`)

```
gradient check: 681 entries, worst relative error 1.672e-06
overfit check: first loss 2.4076 -> last loss 0.0579
causality check: future-token leakage probe passed
all tests passed
```

The gradient check compares every analytic gradient against numerical
gradients; the overfit check proves the model can memorize a tiny batch;
the causality check proves predictions never attend to future tokens.
