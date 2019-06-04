from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?export=download&id=1hVcNufZnHhyoC4c-VciSY-hI7chUU5aj'
export_file_name = 'model'
# classes = ['Action_painting',
#  'Analytical_Cubism',
#  'Art_Nouveau_Modern',
#  'Baroque',
#  'Color_Field_Painting',
#  'Contemporary_Realism',
#  'Cubism',
#  'Early_Renaissance',
#  'Expressionism',
#  'Fauvism',
#  'Impressionism',
#  'Mannerism_Late_Renaissance',
#  'Minimalism',
#  'Naive_Art_Primitivism',
#  'New_Realism',
#  'Northern_Renaissance',
#  'Pointillism',
#  'Pop_Art',
#  'Post_Impressionism',
#  'Realism',
#  'Rococo',
#  'Romanticism',
#  'Ukiyo_e']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    # data_bunch = ImageDataBunch.single_from_classes(path, classes,
    #     tfms=get_transforms(), size=224).normalize(imagenet_stats)
    # learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn = load_learner(path, export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': learn.predict(img)[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

