Out of box motion estimation (optical flow, points, etc) models
![result](https://github.com/user-attachments/assets/8a2da45c-b207-45e7-b1c6-11613627f3da)
[SOURCE](https://www.bilibili.com/video/BV1wRfcYwEXL/?spm_id)

## Usage

``` python

from utils.io_utils import load_image
from PIL import Image

from annotator.animerun_flow import apply_gmflow, apply_raft
# or from annotator.flowformer import apply_flow_former

frame1 = load_image('assets/samples/p1.jxl')
frame2 = load_image('assets/samples/p2.jxl')
flow = apply_raft(frame1, frame2)

from annotator.flow_utils import flow2rgb
flow_vis = flow2rgb(flow)

Image.fromarray(flow_vis).save('result.png')

```

## Acknowledgements
https://github.com/drinkingcoder/FlowFormer-Official  
https://github.com/lisiyao21/AnimeRun  


## License

Please refer to corresponding folders in ```annotator```
