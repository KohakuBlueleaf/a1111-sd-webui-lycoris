# a1111-sd-webui-lycoris

An extension for loading lycoris model in sd-webui. 
I made this stand alone extension (Use sd-webui's extra networks api) to avoid some conflict with other loras extensions.

### LyCORIS
https://github.com/KohakuBlueleaf/LyCORIS

### usage
Install it and restart the webui 
**Don't use "Apply and restart UI", please restart the webui process**

And you will find "LyCORIS" tab in the extra networks page
Use `<lyco:MODEL:WEIGHT>` to utilize the lycoris model
![image](https://user-images.githubusercontent.com/59680068/230762416-be1d3712-65f2-4dd1-ac7a-f403c914dd9b.png)

The launch parameter `--lyco-dir` can be used to define LyCORIS models location path

## Arguments
sd-webui use this format to use extra networks: `<TYPE:MODEL_NAME:arg1:arg2:arg3...:argn>`<br>
With more and more different algorithm be implemented into lycoris, the arguments become more.<br>
So I design this arg system to use it more easily(Maybe):<br>
<br>
`<lyco:MODEL:arg1:arg2:k1=v1:k2=v2>`<br>
<br>
For example, we have te/unet/dyn these 3 arguments, if you want te=1, unet=0.5, dyn=13, you can use it like these:<br>
`<lyco:Model:1:0.5:13>`<br>
`<lyco:Model:1:0.5:dyn=13>`<br>
`<lyco:Model:1:unet=0.5:dyn=13>`<br>
And if you specify ALL the key name, you can ignore the order:<br>
(or, actually, we only count the args, no k-v pair, so dyn=13:unet=1:0.5 also work, but 0.5 is for te (the first argument))<br>
`<lyco:Model:dyn=13:te=1:unet=0.5>`<br>
<br>
And since te=1 is default value, you can also do it like this:<br>
`<lyco:Model:unet=0.5:dyn=13>`<br>

And here is the list for arguments:
| Argument    | What it does| default type and value|
| ----------- | ----------- | ----------- |
| te          | the weight for text encoder | `float: 1.0`|
| unet   | the weight for UNet, when it is None, it use same value as te | `float: None`|
| dyn | How many row you want to utilize when using dylora, if you set to 0, it will disable the dylora| `int: None` |
