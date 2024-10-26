import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os
from tqdm import tqdm

# These are the bike image lists from your files
LIST_OF_BICYCLES = [
        "http://images.cocodataset.org/train2017/000000483108.jpg",
        "http://images.cocodataset.org/train2017/000000293802.jpg",
        "http://images.cocodataset.org/train2017/000000079841.jpg",
        "http://images.cocodataset.org/train2017/000000515289.jpg",
        "http://images.cocodataset.org/train2017/000000562150.jpg",
        "http://images.cocodataset.org/train2017/000000412151.jpg",
        "http://images.cocodataset.org/train2017/000000462565.jpg",
        "http://images.cocodataset.org/train2017/000000509822.jpg",
        "http://images.cocodataset.org/train2017/000000321107.jpg",
        "http://images.cocodataset.org/train2017/000000061181.jpg",
        "http://images.cocodataset.org/train2017/000000018783.jpg",
        "http://images.cocodataset.org/train2017/000000012896.jpg",
        "http://images.cocodataset.org/train2017/000000465692.jpg",
        "http://images.cocodataset.org/train2017/000000391584.jpg",
        "http://images.cocodataset.org/train2017/000000241350.jpg",
        "http://images.cocodataset.org/train2017/000000438024.jpg",
        "http://images.cocodataset.org/train2017/000000442726.jpg",
        "http://images.cocodataset.org/train2017/000000435937.jpg",
        "http://images.cocodataset.org/train2017/000000292819.jpg",
        "http://images.cocodataset.org/train2017/000000157416.jpg",
        "http://images.cocodataset.org/train2017/000000010393.jpg",
        "http://images.cocodataset.org/train2017/000000084540.jpg",
        "http://images.cocodataset.org/train2017/000000007125.jpg",
        "http://images.cocodataset.org/train2017/000000507249.jpg",
        "http://images.cocodataset.org/train2017/000000075923.jpg",
        "http://images.cocodataset.org/train2017/000000240918.jpg",
        "http://images.cocodataset.org/train2017/000000122302.jpg",
        "http://images.cocodataset.org/train2017/000000140006.jpg",
        "http://images.cocodataset.org/train2017/000000536444.jpg",
        "http://images.cocodataset.org/train2017/000000344271.jpg",
        "http://images.cocodataset.org/train2017/000000420081.jpg",
        "http://images.cocodataset.org/train2017/000000148668.jpg",
        "http://images.cocodataset.org/train2017/000000390137.jpg",
        "http://images.cocodataset.org/train2017/000000114183.jpg",
        "http://images.cocodataset.org/train2017/000000020307.jpg",
        "http://images.cocodataset.org/train2017/000000280736.jpg",
        "http://images.cocodataset.org/train2017/000000536321.jpg",
        "http://images.cocodataset.org/train2017/000000188146.jpg",
        "http://images.cocodataset.org/train2017/000000559312.jpg",
        "http://images.cocodataset.org/train2017/000000535808.jpg",
        "http://images.cocodataset.org/train2017/000000451944.jpg",
        "http://images.cocodataset.org/train2017/000000212558.jpg",
        "http://images.cocodataset.org/train2017/000000377867.jpg",
        "http://images.cocodataset.org/train2017/000000139291.jpg",
        "http://images.cocodataset.org/train2017/000000456323.jpg",
        "http://images.cocodataset.org/train2017/000000549386.jpg",
        "http://images.cocodataset.org/train2017/000000254491.jpg",
        "http://images.cocodataset.org/train2017/000000314515.jpg",
        "http://images.cocodataset.org/train2017/000000415904.jpg",
        "http://images.cocodataset.org/train2017/000000101636.jpg",
        "http://images.cocodataset.org/train2017/000000315173.jpg",
        "http://images.cocodataset.org/train2017/000000260627.jpg",
        "http://images.cocodataset.org/train2017/000000001722.jpg",
        "http://images.cocodataset.org/train2017/000000031092.jpg",
        "http://images.cocodataset.org/train2017/000000556205.jpg",
        "http://images.cocodataset.org/train2017/000000049097.jpg",
        "http://images.cocodataset.org/train2017/000000070815.jpg",
        "http://images.cocodataset.org/train2017/000000467000.jpg",
        "http://images.cocodataset.org/train2017/000000416733.jpg",
        "http://images.cocodataset.org/train2017/000000203912.jpg",
        "http://images.cocodataset.org/train2017/000000408143.jpg",
        "http://images.cocodataset.org/train2017/000000120340.jpg",
        "http://images.cocodataset.org/train2017/000000124462.jpg",
        "http://images.cocodataset.org/train2017/000000142718.jpg",
        "http://images.cocodataset.org/train2017/000000108838.jpg",
        "http://images.cocodataset.org/train2017/000000445309.jpg",
        "http://images.cocodataset.org/train2017/000000140197.jpg",
        "http://images.cocodataset.org/train2017/000000012993.jpg",
        "http://images.cocodataset.org/train2017/000000111099.jpg",
        "http://images.cocodataset.org/train2017/000000215867.jpg",
        "http://images.cocodataset.org/train2017/000000565085.jpg",
        "http://images.cocodataset.org/train2017/000000314986.jpg",
        "http://images.cocodataset.org/train2017/000000158708.jpg",
        "http://images.cocodataset.org/train2017/000000263961.jpg",
        "http://images.cocodataset.org/train2017/000000192128.jpg",
        "http://images.cocodataset.org/train2017/000000377832.jpg",
        "http://images.cocodataset.org/train2017/000000187286.jpg",
        "http://images.cocodataset.org/train2017/000000195510.jpg",
        "http://images.cocodataset.org/train2017/000000406949.jpg",
        "http://images.cocodataset.org/train2017/000000330455.jpg",
]

LIST_OF_NON_BICYCLES = [
        "http://images.cocodataset.org/train2017/000000522418.jpg",
        "http://images.cocodataset.org/train2017/000000184613.jpg",
        "http://images.cocodataset.org/train2017/000000318219.jpg",
        "http://images.cocodataset.org/train2017/000000554625.jpg",
        "http://images.cocodataset.org/train2017/000000574769.jpg",
        "http://images.cocodataset.org/train2017/000000060623.jpg",
        "http://images.cocodataset.org/train2017/000000309022.jpg",
        "http://images.cocodataset.org/train2017/000000005802.jpg",
        "http://images.cocodataset.org/train2017/000000222564.jpg",
        "http://images.cocodataset.org/train2017/000000118113.jpg",
        "http://images.cocodataset.org/train2017/000000193271.jpg",
        "http://images.cocodataset.org/train2017/000000224736.jpg",
        "http://images.cocodataset.org/train2017/000000403013.jpg",
        "http://images.cocodataset.org/train2017/000000374628.jpg",
        "http://images.cocodataset.org/train2017/000000328757.jpg",
        "http://images.cocodataset.org/train2017/000000384213.jpg",
        "http://images.cocodataset.org/train2017/000000086408.jpg",
        "http://images.cocodataset.org/train2017/000000372938.jpg",
        "http://images.cocodataset.org/train2017/000000386164.jpg",
        "http://images.cocodataset.org/train2017/000000223648.jpg",
        "http://images.cocodataset.org/train2017/000000204805.jpg",
        "http://images.cocodataset.org/train2017/000000113588.jpg",
        "http://images.cocodataset.org/train2017/000000384553.jpg",
        "http://images.cocodataset.org/train2017/000000337264.jpg",
        "http://images.cocodataset.org/train2017/000000368402.jpg",
        "http://images.cocodataset.org/train2017/000000012448.jpg",
        "http://images.cocodataset.org/train2017/000000542145.jpg",
        "http://images.cocodataset.org/train2017/000000540186.jpg",
        "http://images.cocodataset.org/train2017/000000242611.jpg",
        "http://images.cocodataset.org/train2017/000000051191.jpg",
        "http://images.cocodataset.org/train2017/000000269105.jpg",
        "http://images.cocodataset.org/train2017/000000294832.jpg",
        "http://images.cocodataset.org/train2017/000000144941.jpg",
        "http://images.cocodataset.org/train2017/000000173350.jpg",
        "http://images.cocodataset.org/train2017/000000060760.jpg",
        "http://images.cocodataset.org/train2017/000000324266.jpg",
        "http://images.cocodataset.org/train2017/000000166532.jpg",
        "http://images.cocodataset.org/train2017/000000262284.jpg",
        "http://images.cocodataset.org/train2017/000000360772.jpg",
        "http://images.cocodataset.org/train2017/000000191381.jpg",
        "http://images.cocodataset.org/train2017/000000111076.jpg",
        "http://images.cocodataset.org/train2017/000000340559.jpg",
        "http://images.cocodataset.org/train2017/000000258985.jpg",
        "http://images.cocodataset.org/train2017/000000229643.jpg",
        "http://images.cocodataset.org/train2017/000000125059.jpg",
        "http://images.cocodataset.org/train2017/000000455483.jpg",
        "http://images.cocodataset.org/train2017/000000436141.jpg",
        "http://images.cocodataset.org/train2017/000000129001.jpg",
        "http://images.cocodataset.org/train2017/000000232262.jpg",
        "http://images.cocodataset.org/train2017/000000166323.jpg",
        "http://images.cocodataset.org/train2017/000000580041.jpg",
        "http://images.cocodataset.org/train2017/000000326781.jpg",
        "http://images.cocodataset.org/train2017/000000387362.jpg",
        "http://images.cocodataset.org/train2017/000000138079.jpg",
        "http://images.cocodataset.org/train2017/000000556616.jpg",
        "http://images.cocodataset.org/train2017/000000472621.jpg",
        "http://images.cocodataset.org/train2017/000000192440.jpg",
        "http://images.cocodataset.org/train2017/000000086320.jpg",
        "http://images.cocodataset.org/train2017/000000256668.jpg",
        "http://images.cocodataset.org/train2017/000000383445.jpg",
        "http://images.cocodataset.org/train2017/000000565797.jpg",
        "http://images.cocodataset.org/train2017/000000081922.jpg",
        "http://images.cocodataset.org/train2017/000000050125.jpg",
        "http://images.cocodataset.org/train2017/000000364521.jpg",
        "http://images.cocodataset.org/train2017/000000394892.jpg",
        "http://images.cocodataset.org/train2017/000000001146.jpg",
        "http://images.cocodataset.org/train2017/000000310391.jpg",
        "http://images.cocodataset.org/train2017/000000097434.jpg",
        "http://images.cocodataset.org/train2017/000000463836.jpg",
        "http://images.cocodataset.org/train2017/000000241876.jpg",
        "http://images.cocodataset.org/train2017/000000156832.jpg",
        "http://images.cocodataset.org/train2017/000000270721.jpg",
        "http://images.cocodataset.org/train2017/000000462341.jpg",
        "http://images.cocodataset.org/train2017/000000310103.jpg",
        "http://images.cocodataset.org/train2017/000000032992.jpg",
        "http://images.cocodataset.org/train2017/000000122851.jpg",
        "http://images.cocodataset.org/train2017/000000540763.jpg",
        "http://images.cocodataset.org/train2017/000000138246.jpg",
        "http://images.cocodataset.org/train2017/000000197254.jpg",
        "http://images.cocodataset.org/train2017/000000032907.jpg",
]

LIST_VALIDATION_BICYCLES = [
    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "http://images.cocodataset.org/val2017/000000174482.jpg",
    "http://images.cocodataset.org/val2017/000000296649.jpg",
    "http://images.cocodataset.org/val2017/000000301135.jpg",
    "http://images.cocodataset.org/val2017/000000356387.jpg",
    "http://images.cocodataset.org/val2017/000000038829.jpg",
    "http://images.cocodataset.org/val2017/000000361103.jpg",
    "http://images.cocodataset.org/val2017/000000403565.jpg",
    "http://images.cocodataset.org/val2017/000000441586.jpg",
    "http://images.cocodataset.org/val2017/000000101762.jpg",
    "http://images.cocodataset.org/val2017/000000203317.jpg",
    "http://images.cocodataset.org/val2017/000000074058.jpg",
    "http://images.cocodataset.org/val2017/000000210273.jpg",
    "http://images.cocodataset.org/val2017/000000169996.jpg",
    "http://images.cocodataset.org/val2017/000000226417.jpg",
    "http://images.cocodataset.org/val2017/000000274687.jpg",
    "http://images.cocodataset.org/val2017/000000010363.jpg",
    "http://images.cocodataset.org/val2017/000000476704.jpg",
    "http://images.cocodataset.org/val2017/000000138639.jpg",
    "http://images.cocodataset.org/val2017/000000139099.jpg",
    "http://images.cocodataset.org/val2017/000000152771.jpg",
    "http://images.cocodataset.org/val2017/000000309391.jpg",
    "http://images.cocodataset.org/val2017/000000432553.jpg",
    "http://images.cocodataset.org/val2017/000000196843.jpg",
    "http://images.cocodataset.org/val2017/000000045596.jpg",
    "http://images.cocodataset.org/val2017/000000070774.jpg",
    "http://images.cocodataset.org/val2017/000000193926.jpg",
    "http://images.cocodataset.org/val2017/000000319607.jpg",
    "http://images.cocodataset.org/val2017/000000492077.jpg",
    "http://images.cocodataset.org/val2017/000000357737.jpg",
    "http://images.cocodataset.org/val2017/000000162858.jpg",
    "http://images.cocodataset.org/val2017/000000076416.jpg",
    "http://images.cocodataset.org/val2017/000000508370.jpg",
    "http://images.cocodataset.org/val2017/000000577932.jpg",
    "http://images.cocodataset.org/val2017/000000144333.jpg",
    "http://images.cocodataset.org/val2017/000000261888.jpg",
    "http://images.cocodataset.org/val2017/000000135670.jpg",
    "http://images.cocodataset.org/val2017/000000011149.jpg",
    "http://images.cocodataset.org/val2017/000000259640.jpg",
    "http://images.cocodataset.org/val2017/000000008211.jpg",
    "http://images.cocodataset.org/val2017/000000306136.jpg",
    "http://images.cocodataset.org/val2017/000000228436.jpg",
    "http://images.cocodataset.org/val2017/000000488592.jpg",
    "http://images.cocodataset.org/val2017/000000350023.jpg",
    "http://images.cocodataset.org/val2017/000000490936.jpg",
    "http://images.cocodataset.org/val2017/000000350122.jpg",
    "http://images.cocodataset.org/val2017/000000295809.jpg",
    "http://images.cocodataset.org/val2017/000000424162.jpg",
    "http://images.cocodataset.org/val2017/000000242287.jpg",
    "http://images.cocodataset.org/val2017/000000259830.jpg",
    "http://images.cocodataset.org/val2017/000000492937.jpg",
    "http://images.cocodataset.org/val2017/000000570834.jpg",
    "http://images.cocodataset.org/val2017/000000224051.jpg",
    "http://images.cocodataset.org/val2017/000000061108.jpg",
    "http://images.cocodataset.org/val2017/000000395180.jpg",
    "http://images.cocodataset.org/val2017/000000472623.jpg",
    "http://images.cocodataset.org/val2017/000000007386.jpg",
    "http://images.cocodataset.org/val2017/000000210299.jpg",
    "http://images.cocodataset.org/val2017/000000109055.jpg",
    "http://images.cocodataset.org/val2017/000000571857.jpg",
    "http://images.cocodataset.org/val2017/000000507037.jpg",
    "http://images.cocodataset.org/val2017/000000251140.jpg",
    "http://images.cocodataset.org/val2017/000000142324.jpg",
    "http://images.cocodataset.org/val2017/000000097988.jpg",
    "http://images.cocodataset.org/val2017/000000338625.jpg",
    "http://images.cocodataset.org/val2017/000000301376.jpg",
    "http://images.cocodataset.org/val2017/000000289343.jpg",
    "http://images.cocodataset.org/val2017/000000256941.jpg",
    "http://images.cocodataset.org/val2017/000000279278.jpg",
    "http://images.cocodataset.org/val2017/000000426166.jpg",
    "http://images.cocodataset.org/val2017/000000388258.jpg",
    "http://images.cocodataset.org/val2017/000000291634.jpg",
    "http://images.cocodataset.org/val2017/000000122166.jpg",
    "http://images.cocodataset.org/val2017/000000414510.jpg",
    "http://images.cocodataset.org/val2017/000000266400.jpg",
    "http://images.cocodataset.org/val2017/000000254814.jpg",
    "http://images.cocodataset.org/val2017/000000370208.jpg",
    "http://images.cocodataset.org/val2017/000000055022.jpg",
    "http://images.cocodataset.org/val2017/000000013177.jpg",
    "http://images.cocodataset.org/val2017/000000531134.jpg",
    "http://images.cocodataset.org/val2017/000000429109.jpg",
    "http://images.cocodataset.org/val2017/000000008899.jpg",
    "http://images.cocodataset.org/val2017/000000343561.jpg",
]

LIST_VALIDATION_NON_BICICYLES = [
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "http://images.cocodataset.org/val2017/000000403385.jpg",
    "http://images.cocodataset.org/val2017/000000006818.jpg",
    "http://images.cocodataset.org/val2017/000000480985.jpg",
    "http://images.cocodataset.org/val2017/000000458054.jpg",
    "http://images.cocodataset.org/val2017/000000331352.jpg",
    "http://images.cocodataset.org/val2017/000000386912.jpg",
    "http://images.cocodataset.org/val2017/000000502136.jpg",
    "http://images.cocodataset.org/val2017/000000491497.jpg",
    "http://images.cocodataset.org/val2017/000000184791.jpg",
    "http://images.cocodataset.org/val2017/000000348881.jpg",
    "http://images.cocodataset.org/val2017/000000289393.jpg",
    "http://images.cocodataset.org/val2017/000000522713.jpg",
    "http://images.cocodataset.org/val2017/000000181666.jpg",
    "http://images.cocodataset.org/val2017/000000017627.jpg",
    "http://images.cocodataset.org/val2017/000000143931.jpg",
    "http://images.cocodataset.org/val2017/000000303818.jpg",
    "http://images.cocodataset.org/val2017/000000463730.jpg",
    "http://images.cocodataset.org/val2017/000000460347.jpg",
    "http://images.cocodataset.org/val2017/000000322864.jpg",
    "http://images.cocodataset.org/val2017/000000226111.jpg",
    "http://images.cocodataset.org/val2017/000000153299.jpg",
    "http://images.cocodataset.org/val2017/000000308394.jpg",
    "http://images.cocodataset.org/val2017/000000456496.jpg",
    "http://images.cocodataset.org/val2017/000000058636.jpg",
    "http://images.cocodataset.org/val2017/000000041888.jpg",
    "http://images.cocodataset.org/val2017/000000184321.jpg",
    "http://images.cocodataset.org/val2017/000000565778.jpg",
    "http://images.cocodataset.org/val2017/000000297343.jpg",
    "http://images.cocodataset.org/val2017/000000336587.jpg",
    "http://images.cocodataset.org/val2017/000000122745.jpg",
    "http://images.cocodataset.org/val2017/000000219578.jpg",
    "http://images.cocodataset.org/val2017/000000555705.jpg",
    "http://images.cocodataset.org/val2017/000000443303.jpg",
    "http://images.cocodataset.org/val2017/000000500663.jpg",
    "http://images.cocodataset.org/val2017/000000418281.jpg",
    "http://images.cocodataset.org/val2017/000000025560.jpg",
    "http://images.cocodataset.org/val2017/000000403817.jpg",
    "http://images.cocodataset.org/val2017/000000085329.jpg",
    "http://images.cocodataset.org/val2017/000000329323.jpg",
    "http://images.cocodataset.org/val2017/000000239274.jpg",
    "http://images.cocodataset.org/val2017/000000286994.jpg",
    "http://images.cocodataset.org/val2017/000000511321.jpg",
    "http://images.cocodataset.org/val2017/000000314294.jpg",
    "http://images.cocodataset.org/val2017/000000233771.jpg",
    "http://images.cocodataset.org/val2017/000000475779.jpg",
    "http://images.cocodataset.org/val2017/000000301867.jpg",
    "http://images.cocodataset.org/val2017/000000312421.jpg",
    "http://images.cocodataset.org/val2017/000000185250.jpg",
    "http://images.cocodataset.org/val2017/000000356427.jpg",
    "http://images.cocodataset.org/val2017/000000572517.jpg",
    "http://images.cocodataset.org/val2017/000000270244.jpg",
    "http://images.cocodataset.org/val2017/000000516316.jpg",
    "http://images.cocodataset.org/val2017/000000125211.jpg",
    "http://images.cocodataset.org/val2017/000000562121.jpg",
    "http://images.cocodataset.org/val2017/000000360661.jpg",
    "http://images.cocodataset.org/val2017/000000016228.jpg",
    "http://images.cocodataset.org/val2017/000000382088.jpg",
    "http://images.cocodataset.org/val2017/000000266409.jpg",
    "http://images.cocodataset.org/val2017/000000430961.jpg",
    "http://images.cocodataset.org/val2017/000000080671.jpg",
    "http://images.cocodataset.org/val2017/000000577539.jpg",
    "http://images.cocodataset.org/val2017/000000104612.jpg",
    "http://images.cocodataset.org/val2017/000000476258.jpg",
    "http://images.cocodataset.org/val2017/000000448365.jpg",
    "http://images.cocodataset.org/val2017/000000035197.jpg",
    "http://images.cocodataset.org/val2017/000000349860.jpg",
    "http://images.cocodataset.org/val2017/000000180135.jpg",
    "http://images.cocodataset.org/val2017/000000486438.jpg",
    "http://images.cocodataset.org/val2017/000000400573.jpg",
    "http://images.cocodataset.org/val2017/000000109798.jpg",
    "http://images.cocodataset.org/val2017/000000370677.jpg",
    "http://images.cocodataset.org/val2017/000000238866.jpg",
    "http://images.cocodataset.org/val2017/000000369370.jpg",
    "http://images.cocodataset.org/val2017/000000502737.jpg",
    "http://images.cocodataset.org/val2017/000000515579.jpg",
    "http://images.cocodataset.org/val2017/000000515445.jpg",
    "http://images.cocodataset.org/val2017/000000173383.jpg",
    "http://images.cocodataset.org/val2017/000000438862.jpg",
    "http://images.cocodataset.org/val2017/000000180560.jpg",
    "http://images.cocodataset.org/val2017/000000347693.jpg",
    "http://images.cocodataset.org/val2017/000000039956.jpg",
]

class BicycleDataset(Dataset):
    def __init__(self, transform=None, train=True):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.train = train
        
        # Use all training data for training, and validation lists for testing
        if train:
            # Add all training bicycle images
            self.image_paths.extend(LIST_OF_BICYCLES)
            self.labels.extend([1] * len(LIST_OF_BICYCLES))
            
            # Add all training non-bicycle images
            self.image_paths.extend(LIST_OF_NON_BICYCLES)
            self.labels.extend([0] * len(LIST_OF_NON_BICYCLES))
        else:
            # Use validation sets
            self.image_paths.extend(LIST_VALIDATION_BICYCLES)
            self.labels.extend([1] * len(LIST_VALIDATION_BICYCLES))
            self.image_paths.extend(LIST_VALIDATION_NON_BICICYLES)  # Note: there's a typo in the original list name
            self.labels.extend([0] * len(LIST_VALIDATION_NON_BICICYLES))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        url = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except:
            # Return a random different index if this one fails
            return self.__getitem__(np.random.randint(0, len(self)))

class BicycleClassifier(nn.Module):
    def __init__(self):
        super(BicycleClassifier, self).__init__()
        # Use EfficientNet-b0 as backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Replace classifier
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        scheduler.step(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}')
        print(f'Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_bicycle_model.pth')

def prepare_model():
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Just resize and normalize for validation/testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Prepare datasets
    train_dataset = BicycleDataset(transform=train_transform, train=True)
    test_dataset = BicycleDataset(transform=test_transform, train=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BicycleClassifier().to(device)

    # Train model
    train_model(model, train_loader, test_loader, device)

    return model

def classify_bicycle(image):
    """Function to be used with run_eval"""
    # Load model if it exists, otherwise train it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BicycleClassifier().to(device)
    
    if os.path.exists('best_bicycle_model.pth'):
        model.load_state_dict(torch.load('best_bicycle_model.pth', map_location=device))
    else:
        model = prepare_model()
    
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert numpy image to tensor
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).float().item()
    
    return int(prediction)

if __name__ == "__main__":
    # Train the model
    model = prepare_model()
