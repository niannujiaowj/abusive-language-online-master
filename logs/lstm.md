# Setting_1:
| Hyper-parameter | value |
|-----------------|-------|
| batch_size      | 64    |
| keep_prob       | 0.5   |
| learning_rate   | 0.1   |
| hidden unites   | 100   |
| sentence size   | 124   |

## Task 1 toxicity
```
Epoch 1, loss=0.49, train accuracy = [0.84292313], test accuracy = [0.8637655], test fscore = [0.45606761], test auc= [0.65876165]
Epoch 2, loss=0.36, train accuracy = [0.89089587], test accuracy = [0.89721074], test fscore = [0.65245207], test auc= [0.70193986]
Epoch 3, loss=0.33, train accuracy = [0.90522354], test accuracy = [0.90043905], test fscore = [0.6355655], test auc= [0.72879962]
Epoch 4, loss=0.31, train accuracy = [0.91088333], test accuracy = [0.90637913], test fscore = [0.68614365], test auc= [0.74416566]
Epoch 5, loss=0.30, train accuracy = [0.91621521], test accuracy = [0.90689566], test fscore = [0.68093189], test auc= [0.75590856]
Epoch 6, loss=0.29, train accuracy = [0.91869583], test accuracy = [0.90431302], test fscore = [0.70444036], test auc= [0.76540986]
Epoch 7, loss=0.28, train accuracy = [0.92308679], test accuracy = [0.91076963], test fscore = [0.71813505], test auc= [0.77450782]
Epoch 8, loss=0.27, train accuracy = [0.92482607], test accuracy = [0.90909091], test fscore = [0.70424228], test auc= [0.78058044]
Epoch 9, loss=0.26, train accuracy = [0.92925981], test accuracy = [0.90922004], test fscore = [0.7016533], test auc= [0.78448953]
Epoch 10, loss=0.26, train accuracy = [0.9291315], test accuracy = [0.90870351], test fscore = [0.69007141], test auc= [0.78638471]
Epoch 11, loss=0.25, train accuracy = [0.93054288], test accuracy = [0.91102789], test fscore = [0.70921625], test auc= [0.78855675]
Epoch 12, loss=0.25, train accuracy = [0.93122719], test accuracy = [0.90470041], test fscore = [0.69461491], test auc= [0.79081789]
Epoch 13, loss=0.25, train accuracy = [0.93118442], test accuracy = [0.90599174], test fscore = [0.70357659], test auc= [0.79281383]
Epoch 14, loss=0.25, train accuracy = [0.93151232], test accuracy = [0.89979339], test fscore = [0.68738137], test auc= [0.79446588]
Epoch 15, loss=0.25, train accuracy = [0.93346544], test accuracy = [0.90508781], test fscore = [0.69235035], test auc= [0.79583583]
Epoch 16, loss=0.26, train accuracy = [0.93028627], test accuracy = [0.90521694], test fscore = [0.69820379], test auc= [0.79729153]
```


```
Epoch 1, loss=0.51, train accuracy = [0.82999259], test accuracy = [0.84116736], test fscore = [0.30119095], test auc= [0.58765216]
Epoch 2, loss=0.39, train accuracy = [0.8825844], test accuracy = [0.88520145], test fscore = [0.55463682], test auc= [0.63230637]
Epoch 3, loss=0.34, train accuracy = [0.90187329], test accuracy = [0.8978564], test fscore = [0.62821073], test auc= [0.67399774]
Epoch 4, loss=0.31, train accuracy = [0.91322137], test accuracy = [0.90754132], test fscore = [0.67590099], test auc= [0.70198683]
Epoch 5, loss=0.29, train accuracy = [0.91920906], test accuracy = [0.90728306], test fscore = [0.66506237], test auc= [0.71922781]
Epoch 6, loss=0.28, train accuracy = [0.92264484], test accuracy = [0.90792872], test fscore = [0.70247535], test auc= [0.73301545]
Epoch 7, loss=0.27, train accuracy = [0.92580976], test accuracy = [0.90431302], test fscore = [0.71177296], test auc= [0.74710821]
Epoch 8, loss=0.26, train accuracy = [0.92749202], test accuracy = [0.90625], test fscore = [0.71720968], test auc= [0.75825324]
Epoch 9, loss=0.26, train accuracy = [0.92809078], test accuracy = [0.91296488], test fscore = [0.71695813], test auc= [0.76630547]
Epoch 10, loss=0.25, train accuracy = [0.93122719], test accuracy = [0.90289256], test fscore = [0.70728148], test auc= [0.77283099]
Epoch 11, loss=0.25, train accuracy = [0.93198278], test accuracy = [0.91206095], test fscore = [0.71187371], test auc= [0.77718215]
Epoch 12, loss=0.25, train accuracy = [0.93278114], test accuracy = [0.90521694], test fscore = [0.71308058], test auc= [0.78103259]
Epoch 13, loss=0.25, train accuracy = [0.93423529], test accuracy = [0.90715393], test fscore = [0.69547839], test auc= [0.7840771]
Epoch 14, loss=0.24, train accuracy = [0.93529026], test accuracy = [0.9058626], test fscore = [0.68361387], test auc= [0.78556672]
Epoch 15, loss=0.25, train accuracy = [0.93241047], test accuracy = [0.90896178], test fscore = [0.69132135], test auc= [0.78636113]
Epoch 16, loss=0.25, train accuracy = [0.93359375], test accuracy = [0.90612087], test fscore = [0.69065505], test auc= [0.78735692]
Epoch 17, loss=0.24, train accuracy = [0.93424954], test accuracy = [0.90689566], test fscore = [0.70418939], test auc= [0.78906534]
Epoch 18, loss=0.25, train accuracy = [0.93282391], test accuracy = [0.91038223], test fscore = [0.71374885], test auc= [0.7910114]
Epoch 19, loss=0.25, train accuracy = [0.93166914], test accuracy = [0.89721074], test fscore = [0.68984348], test auc= [0.7927748]
Epoch 20, loss=0.25, train accuracy = [0.93245324], test accuracy = [0.89746901], test fscore = [0.69285569], test auc= [0.79429273]
Epoch 21, loss=0.25, train accuracy = [0.93453467], test accuracy = [0.90418388], test fscore = [0.70374804], test auc= [0.79570461]
Epoch 22, loss=0.25, train accuracy = [0.93313755], test accuracy = [0.90134298], test fscore = [0.6963313], test auc= [0.79696107]
Epoch 23, loss=0.25, train accuracy = [0.93366503], test accuracy = [0.90883264], test fscore = [0.70507338], test auc= [0.79799999]
Epoch 24, loss=0.26, train accuracy = [0.93335139], test accuracy = [0.90173037], test fscore = [0.68964779], test auc= [0.79878235]
Epoch 25, loss=0.25, train accuracy = [0.93169765], test accuracy = [0.89682335], test fscore = [0.68857101], test auc= [0.79969044]
Epoch 26, loss=0.25, train accuracy = [0.93440636], test accuracy = [0.90650826], test fscore = [0.68519052], test auc= [0.79998351]
Epoch 27, loss=0.26, train accuracy = [0.93229642], test accuracy = [0.90405475], test fscore = [0.67920292], test auc= [0.80002487]
Epoch 28, loss=0.26, train accuracy = [0.92821909], test accuracy = [0.89230372], test fscore = [0.64824184], test auc= [0.79961187]
Epoch 29, loss=0.27, train accuracy = [0.92729243], test accuracy = [0.90108471], test fscore = [0.68791787], test auc= [0.79957587]
Epoch 30, loss=0.26, train accuracy = [0.92676494], test accuracy = [0.89243285], test fscore = [0.68477197], test auc= [0.80020533]
Epoch 31, loss=0.26, train accuracy = [0.92944514], test accuracy = [0.89733988], test fscore = [0.6889079], test auc= [0.80082387]
Epoch 32, loss=0.27, train accuracy = [0.92841868], test accuracy = [0.90340909], test fscore = [0.68972445], test auc= [0.80104065]
```

## Task 2 aggression
```
Epoch 1, loss=0.53, train accuracy = [0.82496008], test accuracy = [0.81443698], test fscore = [0.], test auc= [0.5]
Epoch 2, loss=0.46, train accuracy = [0.8427378], test accuracy = [0.87086777], test fscore = [0.51808775], test auc= [0.56370217]
Epoch 3, loss=0.38, train accuracy = [0.87862112], test accuracy = [0.87758264], test fscore = [0.63837997], test auc= [0.63789021]
Epoch 4, loss=0.35, train accuracy = [0.88915659], test accuracy = [0.89204545], test fscore = [0.6186137], test auc= [0.67620364]
Epoch 5, loss=0.33, train accuracy = [0.89832345], test accuracy = [0.89888946], test fscore = [0.66397502], test auc= [0.69636422]
Epoch 6, loss=0.32, train accuracy = [0.9044537], test accuracy = [0.89359504], test fscore = [0.6731677], test auc= [0.71327455]
Epoch 7, loss=0.31, train accuracy = [0.90848825], test accuracy = [0.8982438], test fscore = [0.69310014], test auc= [0.72807009]
Epoch 8, loss=0.30, train accuracy = [0.91364906], test accuracy = [0.90198864], test fscore = [0.6865709], test auc= [0.7384689]
Epoch 9, loss=0.30, train accuracy = [0.91478958], test accuracy = [0.89527376], test fscore = [0.68682524], test auc= [0.74681306]
Epoch 10, loss=0.29, train accuracy = [0.91805429], test accuracy = [0.88597624], test fscore = [0.67644541], test auc= [0.75424421]
Epoch 11, loss=0.29, train accuracy = [0.91956547], test accuracy = [0.89927686], test fscore = [0.65729675], test auc= [0.75824719]
Epoch 12, loss=0.28, train accuracy = [0.9194229], test accuracy = [0.8950155], test fscore = [0.66808169], test auc= [0.76066722]
Epoch 13, loss=0.28, train accuracy = [0.92157562], test accuracy = [0.88855888], test fscore = [0.67324377], test auc= [0.76417415]
Epoch 14, loss=0.28, train accuracy = [0.92308679], test accuracy = [0.89953512], test fscore = [0.67459782], test auc= [0.76714193]
Epoch 15, loss=0.29, train accuracy = [0.92139028], test accuracy = [0.89139979], test fscore = [0.6580883], test auc= [0.76926466]
Epoch 16, loss=0.30, train accuracy = [0.91980782], test accuracy = [0.89204545], test fscore = [0.65068444], test auc= [0.77036557]
Epoch 17, loss=0.29, train accuracy = [0.91277943], test accuracy = [0.89307851], test fscore = [0.65261714], test auc= [0.77089407]
Epoch 18, loss=0.30, train accuracy = [0.9180828], test accuracy = [0.89385331], test fscore = [0.66624004], test auc= [0.7718649]
Epoch 19, loss=0.28, train accuracy = [0.92137603], test accuracy = [0.890625], test fscore = [0.66573503], test auc= [0.77328365]
Epoch 20, loss=0.28, train accuracy = [0.92267336], test accuracy = [0.89630682], test fscore = [0.67643956], test auc= [0.77444319]
Epoch 21, loss=0.29, train accuracy = [0.91444742], test accuracy = [0.89152893], test fscore = [0.66651363], test auc= [0.77565854]
Epoch 22, loss=0.31, train accuracy = [0.91523152], test accuracy = [0.89333678], test fscore = [0.66262006], test auc= [0.77652646]
Epoch 23, loss=0.32, train accuracy = [0.90733349], test accuracy = [0.89049587], test fscore = [0.64637028], test auc= [0.77695661]
Epoch 24, loss=0.32, train accuracy = [0.90401175], test accuracy = [0.890625], test fscore = [0.67419489], test auc= [0.77762326]
Epoch 25, loss=0.31, train accuracy = [0.91370609], test accuracy = [0.89139979], test fscore = [0.66267931], test auc= [0.77862227]
Epoch 26, loss=0.31, train accuracy = [0.91578752], test accuracy = [0.89540289], test fscore = [0.65194761], test auc= [0.77894025]
Epoch 27, loss=0.31, train accuracy = [0.91805429], test accuracy = [0.89385331], test fscore = [0.6661637], test auc= [0.7792528]
Epoch 28, loss=0.29, train accuracy = [0.91702783], test accuracy = [0.89540289], test fscore = [0.65119357], test auc= [0.77944297]
Epoch 29, loss=0.32, train accuracy = [0.91058394], test accuracy = [0.8910124], test fscore = [0.64762327], test auc= [0.77935936]
Epoch 30, loss=0.31, train accuracy = [0.91025604], test accuracy = [0.89449897], test fscore = [0.63195913], test auc= [0.77900771]
Epoch 31, loss=0.31, train accuracy = [0.91240876], test accuracy = [0.8866219], test fscore = [0.66414275], test auc= [0.77928616]
Epoch 32, loss=0.31, train accuracy = [0.9135065], test accuracy = [0.8866219], test fscore = [0.65443642], test auc= [0.77985697]
```

## Task 3 logs


# Run 2
## Task 1 logs
```
Epoch 0, loss=0.507905, train accuracy = [0.83282961], test accuracy = [0.85459711], test p = [0.22812447], test r = [0.80425616], test fscore = [0.34061327], test auc= [0.61311517]
Epoch 1, loss=0.397966, train accuracy = [0.87512831], test accuracy = [0.88029442], test p = [0.52281653], test r = [0.73273183], test fscore = [0.59489249], test auc= [0.65562521]
Epoch 2, loss=0.351786, train accuracy = [0.89545792], test accuracy = [0.89346591], test p = [0.65469115], test r = [0.71425108], test fscore = [0.67229019], test auc= [0.70392382]
Epoch 3, loss=0.323068, train accuracy = [0.90729072], test accuracy = [0.89669421], test p = [0.65940969], test r = [0.72629646], test fscore = [0.68035688], test auc= [0.73321495]
Epoch 4, loss=0.307670, train accuracy = [0.91218066], test accuracy = [0.9058626], test p = [0.58465239], test r = [0.83322613], test fscore = [0.67496648], test auc= [0.74706817]
Epoch 5, loss=0.299440, train accuracy = [0.91721316], test accuracy = [0.89591942], test p = [0.72558759], test r = [0.6887331], test fscore = [0.70057478], test auc= [0.75891271]
Epoch 6, loss=0.287277, train accuracy = [0.92042085], test accuracy = [0.90728306], test p = [0.67015884], test r = [0.77105339], test fscore = [0.70763218], test auc= [0.76918585]
Epoch 7, loss=0.277450, train accuracy = [0.92382812], test accuracy = [0.90844525], test p = [0.67456727], test r = [0.77134285], test fscore = [0.70908736], test auc= [0.77599536]
Epoch 8, loss=0.268648, train accuracy = [0.92676494], test accuracy = [0.89282025], test p = [0.75187558], test r = [0.67411606], test fscore = [0.7015758], test auc= [0.7827185]
Epoch 9, loss=0.268688, train accuracy = [0.92927406], test accuracy = [0.90018079], test p = [0.71040846], test r = [0.71406913], test fscore = [0.70292775], test auc= [0.78825341]
Epoch 10, loss=0.268426, train accuracy = [0.92870381], test accuracy = [0.90495868], test p = [0.71639314], test r = [0.72968514], test fscore = [0.71346895], test auc= [0.79255033]
Epoch 11, loss=0.260488, train accuracy = [0.9305714], test accuracy = [0.90547521], test p = [0.65593762], test r = [0.76905829], test fscore = [0.69624133], test auc= [0.79515605]
Epoch 12, loss=0.256113, train accuracy = [0.93152657], test accuracy = [0.90599174], test p = [0.68839592], test r = [0.74304139], test fscore = [0.70546118], test auc= [0.79729109]
Epoch 13, loss=0.268188, train accuracy = [0.9272354], test accuracy = [0.89527376], test p = [0.48277081], test r = [0.87353055], test fscore = [0.60714559], test auc= [0.79575896]
Epoch 14, loss=0.258017, train accuracy = [0.92916001], test accuracy = [0.89708161], test p = [0.70904493], test r = [0.7055712], test fscore = [0.69850079], test auc= [0.79527377]
Epoch 15, loss=0.257854, train accuracy = [0.92997263], test accuracy = [0.90043905], test p = [0.66264805], test r = [0.73991711], test fscore = [0.69039854], test auc= [0.79685173]
Epoch 16, loss=0.252665, train accuracy = [0.93164062], test accuracy = [0.90560434], test p = [0.67422021], test r = [0.75984526], test fscore = [0.70518759], test auc= [0.79792079]
Epoch 17, loss=0.256399, train accuracy = [0.93035755], test accuracy = [0.89566116], test p = [0.69396403], test r = [0.69953847], test fscore = [0.68909803], test auc= [0.79922624]
Epoch 18, loss=0.254286, train accuracy = [0.93272411], test accuracy = [0.90728306], test p = [0.63924728], test r = [0.79417836], test fscore = [0.69587298], test auc= [0.80000463]
Epoch 19, loss=0.252849, train accuracy = [0.93232493], test accuracy = [0.90470041], test p = [0.66782056], test r = [0.76427454], test fscore = [0.70256246], test auc= [0.80054259]
Epoch 20, loss=0.256263, train accuracy = [0.93178319], test accuracy = [0.9066374], test p = [0.67712013], test r = [0.7678798], test fscore = [0.70906437], test auc= [0.80145648]
Epoch 21, loss=0.247691, train accuracy = [0.93312329], test accuracy = [0.90069731], test p = [0.70573253], test r = [0.71986893], test fscore = [0.70215025], test auc= [0.80255512]
Epoch 22, loss=0.257062, train accuracy = [0.93105611], test accuracy = [0.90082645], test p = [0.65712764], test r = [0.7450134], test fscore = [0.6894309], test auc= [0.80322783]
Epoch 23, loss=0.256304, train accuracy = [0.93088504], test accuracy = [0.90521694], test p = [0.62588248], test r = [0.79256927], test fscore = [0.68513429], test auc= [0.8032724]
Epoch 24, loss=0.257917, train accuracy = [0.92903171], test accuracy = [0.90457128], test p = [0.63028234], test r = [0.78000416], test fscore = [0.685017], test auc= [0.80324132]
Epoch 25, loss=0.256024, train accuracy = [0.92988709], test accuracy = [0.90818698], test p = [0.6232867], test r = [0.81233034], test fscore = [0.69266181], test auc= [0.80319397]
Epoch 26, loss=0.258519, train accuracy = [0.92901745], test accuracy = [0.89953512], test p = [0.6385152], test r = [0.74846767], test fscore = [0.67875688], test auc= [0.8031417]
Epoch 27, loss=0.264050, train accuracy = [0.9255674], test accuracy = [0.89346591], test p = [0.6913271], test r = [0.70153555], test fscore = [0.68589285], test auc= [0.80348483]
Epoch 28, loss=0.260294, train accuracy = [0.92744925], test accuracy = [0.89837293], test p = [0.68654144], test r = [0.72512122], test fscore = [0.69399093], test auc= [0.80403131]
Epoch 29, loss=0.269950, train accuracy = [0.92695027], test accuracy = [0.89617769], test p = [0.69425264], test r = [0.70064581], test fscore = [0.68980575], test auc= [0.80455982]
Epoch 30, loss=0.280329, train accuracy = [0.92635151], test accuracy = [0.89733988], test p = [0.62647701], test r = [0.75448681], test fscore = [0.67249647], test auc= [0.80460061]
Epoch 31, loss=0.271514, train accuracy = [0.92414177], test accuracy = [0.90650826], test p = [0.66147169], test r = [0.77354027], test fscore = [0.70165703], test auc= [0.8046491]
Epoch 32, loss=0.269135, train accuracy = [0.92652258], test accuracy = [0.90534607], test p = [0.62368536], test r = [0.79610627], test fscore = [0.68819696], test auc= [0.80473302]
training time = 178.693720
```

## Task 2 logs
```
Epoch 0, loss=0.528093, train accuracy = [0.82441834], test accuracy = [0.82502583], test p = [0.12481025], test r = [0.53794398], test fscore = [0.19166844], test auc= [0.54725628]
Epoch 1, loss=0.402611, train accuracy = [0.86934021], test accuracy = [0.87990702], test p = [0.43341407], test r = [0.79881017], test fscore = [0.54535903], test auc= [0.6071126]
Epoch 2, loss=0.356843, train accuracy = [0.88944172], test accuracy = [0.88675103], test p = [0.57548175], test r = [0.72685846], test fscore = [0.62790626], test auc= [0.66080311]
Epoch 3, loss=0.334449, train accuracy = [0.89890796], test accuracy = [0.89656508], test p = [0.61472733], test r = [0.76843736], test fscore = [0.66662673], test auc= [0.69496275]
Epoch 4, loss=0.315234, train accuracy = [0.90492416], test accuracy = [0.8942407], test p = [0.65174917], test r = [0.73827034], test fscore = [0.67835525], test auc= [0.71834041]
Epoch 5, loss=0.306239, train accuracy = [0.90977133], test accuracy = [0.89953512], test p = [0.56731745], test r = [0.81412046], test fscore = [0.65438717], test auc= [0.7303429]
Epoch 6, loss=0.292785, train accuracy = [0.91588732], test accuracy = [0.90056818], test p = [0.63337202], test r = [0.77690461], test fscore = [0.68504418], test auc= [0.73930819]
Epoch 7, loss=0.287321, train accuracy = [0.91745552], test accuracy = [0.89333678], test p = [0.67486213], test r = [0.71175058], test fscore = [0.67997037], test auc= [0.74848938]
Epoch 8, loss=0.280913, train accuracy = [0.92090557], test accuracy = [0.89579029], test p = [0.60051694], test r = [0.76725441], test fscore = [0.65807089], test auc= [0.75444151]
Epoch 9, loss=0.273202, train accuracy = [0.92225992], test accuracy = [0.89669421], test p = [0.62132461], test r = [0.74640711], test fscore = [0.66615276], test auc= [0.75806686]
Epoch 10, loss=0.267183, train accuracy = [0.9261234], test accuracy = [0.89643595], test p = [0.61902159], test r = [0.75250912], test fscore = [0.66760748], test auc= [0.76124907]
Epoch 11, loss=0.264380, train accuracy = [0.92549612], test accuracy = [0.89088326], test p = [0.6341427], test r = [0.71223664], test fscore = [0.65807747], test auc= [0.76439919]
Epoch 12, loss=0.260843, train accuracy = [0.92695027], test accuracy = [0.89514463], test p = [0.6068047], test r = [0.75961499], test fscore = [0.66364036], test auc= [0.7663561]
Epoch 13, loss=0.262211, train accuracy = [0.92535356], test accuracy = [0.89114153], test p = [0.63753471], test r = [0.71858235], test fscore = [0.66538997], test auc= [0.76804238]
Epoch 14, loss=0.261139, train accuracy = [0.92764884], test accuracy = [0.89230372], test p = [0.62653187], test r = [0.72856604], test fscore = [0.66232034], test auc= [0.76995497]
Epoch 15, loss=0.258424, train accuracy = [0.92851848], test accuracy = [0.88997934], test p = [0.63841253], test r = [0.71300385], test fscore = [0.66052783], test auc= [0.77137113]
Epoch 16, loss=0.257594, train accuracy = [0.93030052], test accuracy = [0.89320764], test p = [0.58002508], test r = [0.76640568], test fscore = [0.64664565], test auc= [0.77223155]
Epoch 17, loss=0.273618, train accuracy = [0.9215471], test accuracy = [0.88920455], test p = [0.62032804], test r = [0.72360759], test fscore = [0.65148393], test auc= [0.77267588]
Epoch 18, loss=0.259260, train accuracy = [0.92740648], test accuracy = [0.89217459], test p = [0.64387829], test r = [0.72663751], test fscore = [0.66959738], test auc= [0.77386891]
Epoch 19, loss=0.268567, train accuracy = [0.92525376], test accuracy = [0.88972107], test p = [0.65271364], test r = [0.70762928], test fscore = [0.66687098], test auc= [0.77512372]
Epoch 20, loss=0.259566, train accuracy = [0.92736371], test accuracy = [0.88429752], test p = [0.61787915], test r = [0.70896423], test fscore = [0.64617154], test auc= [0.77586314]
Epoch 21, loss=0.268444, train accuracy = [0.92780566], test accuracy = [0.89127066], test p = [0.5885643], test r = [0.7483587], test fscore = [0.6472651], test auc= [0.77598439]
Epoch 22, loss=0.268709, train accuracy = [0.92512546], test accuracy = [0.88739669], test p = [0.52034304], test r = [0.77338557], test fscore = [0.61098086], test auc= [0.77532612]
Epoch 23, loss=0.263665, train accuracy = [0.92724966], test accuracy = [0.88907541], test p = [0.64321733], test r = [0.707702], test fscore = [0.66331727], test auc= [0.77538091]
Epoch 24, loss=0.265855, train accuracy = [0.92588104], test accuracy = [0.89372417], test p = [0.62120168], test r = [0.7399549], test fscore = [0.6649476], test auc= [0.77618005]
Epoch 25, loss=0.265663, train accuracy = [0.92629448], test accuracy = [0.88817149], test p = [0.59949889], test r = [0.72228435], test fscore = [0.64327524], test auc= [0.77653699]
Epoch 26, loss=0.268304, train accuracy = [0.92526802], test accuracy = [0.89049587], test p = [0.65028084], test r = [0.71832772], test fscore = [0.66760913], test auc= [0.77705118]
Epoch 27, loss=0.294607, train accuracy = [0.91668568], test accuracy = [0.87887397], test p = [0.6259913], test r = [0.66959227], test fscore = [0.63531739], test auc= [0.77752597]
Epoch 28, loss=0.282107, train accuracy = [0.92024977], test accuracy = [0.88894628], test p = [0.66326095], test r = [0.69505368], test fscore = [0.66878492], test auc= [0.77819131]
Epoch 29, loss=0.276711, train accuracy = [0.92211736], test accuracy = [0.88868802], test p = [0.62435858], test r = [0.72675845], test fscore = [0.65492876], test auc= [0.77870222]
Epoch 30, loss=0.281326, train accuracy = [0.92123346], test accuracy = [0.88778409], test p = [0.65876417], test r = [0.69838524], test fscore = [0.66344009], test auc= [0.77924839]
Epoch 31, loss=0.273928, train accuracy = [0.92513971], test accuracy = [0.8942407], test p = [0.65319998], test r = [0.71760975], test fscore = [0.67311036], test auc= [0.78004197]
Epoch 32, loss=0.269894, train accuracy = [0.92502566], test accuracy = [0.89165806], test p = [0.61123863], test r = [0.73557167], test fscore = [0.65540296], test auc= [0.78037193]
training time = 181.617415
```


## Task 3 logs
```
Epoch 0, loss=0.499938, train accuracy = [0.83985801], test accuracy = [0.85123967], test p = [0.36452635], test r = [0.59066354], test fscore = [0.43399596], test auc= [0.64709548]
Epoch 1, loss=0.374268, train accuracy = [0.88664747], test accuracy = [0.8942407], test p = [0.41777157], test r = [0.87469848], test fscore = [0.54162678], test auc= [0.67494239]
Epoch 2, loss=0.329146, train accuracy = [0.90398323], test accuracy = [0.90302169], test p = [0.51142501], test r = [0.84887206], test fscore = [0.61822514], test auc= [0.69655899]
Epoch 3, loss=0.309447, train accuracy = [0.91270814], test accuracy = [0.91012397], test p = [0.62107932], test r = [0.79459412], test fscore = [0.67971497], test auc= [0.71883506]
Epoch 4, loss=0.297505, train accuracy = [0.91843921], test accuracy = [0.91257748], test p = [0.58702383], test r = [0.83302522], test fscore = [0.6711033], test auc= [0.7352928]
Epoch 5, loss=0.288555, train accuracy = [0.92017849], test accuracy = [0.91051136], test p = [0.5738721], test r = [0.83168305], test fscore = [0.66083526], test auc= [0.74386553]
Epoch 6, loss=0.279189, train accuracy = [0.92462648], test accuracy = [0.90805785], test p = [0.52434563], test r = [0.86070358], test fscore = [0.63054877], test auc= [0.7477726]
Epoch 7, loss=0.273005, train accuracy = [0.92762032], test accuracy = [0.90392562], test p = [0.70028083], test r = [0.7130446], test fscore = [0.69200599], test auc= [0.75417378]
Epoch 8, loss=0.270455, train accuracy = [0.92734945], test accuracy = [0.90069731], test p = [0.66671077], test r = [0.71432528], test fscore = [0.67575922], test auc= [0.76169662]
Epoch 9, loss=0.281545, train accuracy = [0.92424156], test accuracy = [0.91412707], test p = [0.64059517], test r = [0.80262998], test fscore = [0.69514192], test auc= [0.76673921]
Epoch 10, loss=0.269950, train accuracy = [0.9293596], test accuracy = [0.91089876], test p = [0.65576912], test r = [0.77539261], test fscore = [0.69502586], test auc= [0.77071576]
Epoch 11, loss=0.267772, train accuracy = [0.92807653], test accuracy = [0.90702479], test p = [0.63954748], test r = [0.75011162], test fscore = [0.67653803], test auc= [0.77394697]
Epoch 12, loss=0.291649, train accuracy = [0.91995039], test accuracy = [0.88274793], test p = [0.60526035], test r = [0.65967007], test fscore = [0.6179079], test auc= [0.77517675]
Epoch 13, loss=0.281616, train accuracy = [0.92371407], test accuracy = [0.90767045], test p = [0.61286134], test r = [0.78961482], test fscore = [0.67217866], test auc= [0.77573149]
Epoch 14, loss=0.267135, train accuracy = [0.92900319], test accuracy = [0.89695248], test p = [0.72074735], test r = [0.67454877], test fscore = [0.68496332], test auc= [0.77815404]
Epoch 15, loss=0.275743, train accuracy = [0.92476905], test accuracy = [0.90844525], test p = [0.6414655], test r = [0.76595195], test fscore = [0.67982776], test auc= [0.78062943]
Epoch 16, loss=0.263434, train accuracy = [0.93081375], test accuracy = [0.90521694], test p = [0.66836721], test r = [0.7393702], test fscore = [0.68400899], test auc= [0.78236912]
Epoch 17, loss=0.261498, train accuracy = [0.93065693], test accuracy = [0.88636364], test p = [0.72468849], test r = [0.63669773], test fscore = [0.66341417], test auc= [0.78459653]
Epoch 18, loss=0.266910, train accuracy = [0.9291315], test accuracy = [0.90676653], test p = [0.64947416], test r = [0.75048883], test fscore = [0.68356256], test auc= [0.7863052]
Epoch 19, loss=0.256668, train accuracy = [0.93248175], test accuracy = [0.91283574], test p = [0.60576261], test r = [0.81539372], test fscore = [0.67685982], test auc= [0.78703295]
Epoch 20, loss=0.261588, train accuracy = [0.93226791], test accuracy = [0.90444215], test p = [0.65652073], test r = [0.73242998], test fscore = [0.67607011], test auc= [0.78766487]
Epoch 21, loss=0.261040, train accuracy = [0.93176893], test accuracy = [0.90276343], test p = [0.67721107], test r = [0.71886136], test fscore = [0.67925211], test auc= [0.78881552]
Epoch 22, loss=0.257701, train accuracy = [0.93443488], test accuracy = [0.88946281], test p = [0.70702535], test r = [0.6506961], test fscore = [0.66408002], test auc= [0.79009808]
Epoch 23, loss=0.255402, train accuracy = [0.93293796], test accuracy = [0.90792872], test p = [0.65785722], test r = [0.75918523], test fscore = [0.68484389], test auc= [0.79109059]
Epoch 24, loss=0.260538, train accuracy = [0.93298073], test accuracy = [0.90650826], test p = [0.64101659], test r = [0.75565241], test fscore = [0.68032165], test auc= [0.79159319]
Epoch 25, loss=0.259593, train accuracy = [0.93149806], test accuracy = [0.90482955], test p = [0.64661153], test r = [0.73689923], test fscore = [0.6741324], test auc= [0.79214093]
Epoch 26, loss=0.255841, train accuracy = [0.9331518], test accuracy = [0.90392562], test p = [0.67330064], test r = [0.70833089], test fscore = [0.67883727], test auc= [0.79293691]
Epoch 27, loss=0.261487, train accuracy = [0.93293796], test accuracy = [0.89656508], test p = [0.6863741], test r = [0.69848371], test fscore = [0.6738565], test auc= [0.79364512]
Epoch 28, loss=0.259233, train accuracy = [0.93355098], test accuracy = [0.90960744], test p = [0.56912483], test r = [0.82824061], test fscore = [0.65724519], test auc= [0.79362513]
Epoch 29, loss=0.261006, train accuracy = [0.93206832], test accuracy = [0.91051136], test p = [0.60125855], test r = [0.81113258], test fscore = [0.66965741], test auc= [0.79336201]
Epoch 30, loss=0.264898, train accuracy = [0.92914576], test accuracy = [0.90560434], test p = [0.62576175], test r = [0.75574408], test fscore = [0.67121626], test auc= [0.79336201]
Epoch 31, loss=0.262225, train accuracy = [0.93360801], test accuracy = [0.89682335], test p = [0.66978733], test r = [0.69499], test fscore = [0.666762], test auc= [0.79374878]
Epoch 32, loss=0.264594, train accuracy = [0.93235344], test accuracy = [0.90947831], test p = [0.6490718], test r = [0.76367927], test fscore = [0.68779692], test auc= [0.79416846]
training time = 179.748662
```
