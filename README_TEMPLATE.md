# Hướng dẫn Project

## Install
`pip install -r requirements.txt`
****bắt buộc***

### config/
Các tham số nào hay thay đổi thì cho vô file `.yaml`. Nếu được thì thêm tiền tố của feature đang làm vô đằng trước để tránh khi merge branch sẽ bị mất, eg `model_config.yaml`, `ui_config.yaml`

### experiments/
Nếu mọi thích code 1 lúc hết pipeline thì vô file `trials.py`, hoặc tạo `.ipynb` để test. Nói chung là thử nghiệm gì thì quăng hết vô đây. Sau khi test okela thì tách thành từng module nhỏ sau cũng được

### tests/
Unit test thôi

### src/english_grading
#### ./config/
Trong `configuration.py`, sẽ có 1 class chung tên là `ConfigurationManager`. Class này sẽ có nhiệm vụ đọc file `*_config.yaml` và lấy ra các config gán vô từng entity liên quan thông qua các method.
```
class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH
    ) -> None:
        # Đọc file yaml
        self.config = read_yaml(config_filepath)

    # Method để lấy config
    def get_database_config(self) -> Database:
        config = self.config.database
        return Database(
            database_path=config.database_path,
            download_url=config.download_url,
            download_id=config.download_id,
            download_path=config.download_path,
            untar_path=config.untar_path,
            host=config.host,
            database=config.database,
            user=config.user,
            password=config.password,
            port=config.port
        )
```

#### ./conponents/
Trong folder này sẽ lưu các feature thành các file `.py`, eg `model_train.py`, `ui.py`.

Trong từng file `.py` là một class tương ứng, class này sẽ nhận config từ method trong `ConfigurationManager`. Class sẽ chứa các method là các bước để chạy feature đó.
> Ví dụ như feature train model thì có các bước như data_ingestion, data_transformation, model_train 
>
> Thì feature đó sẽ có file tên `model_train.py`, có các method tương ứng là 
>
> Class ModelTrain:
> - def __init__(self, config): self.config = config
> - def data_ingestion(self)
> - def data_transformation(self)
> - def model_train(self)

#### ./constants/
Các hằng số (constants) của các feature được lưu thành các file `.py` riêng. Hằng số thì nhớ viết HOA

#### ./entity/
Feature sẽ có các entity, tạo file `.py` rồi tạo class để lưu entity đó. Các entity sẽ lấy tham số từ config, nên config như thế nào thì tạo entity như vậy.

```
model.py

from dataclasses import dataclass
from pathlib import Path

# Entity
@dataclass
class Database: 
    database_path: Path
    download_url: Path
    download_id: Path
    download_path: Path
    untar_path: Path
    # Database info
    host: str
    database: str
    user: str
    password: str
    port: str

```

#### ./logging/
Logging thôi mn không cần đụng. Nếu có cần log gì ra thì trong mấy file kia chỉ cần
```
from english_grading.logging import logger
logger.info("<cái gì đó cần log>")
```

#### ./pipeline/
Từng feature tạo 1 file pipeline riêng, pipeline này sẽ link với `./conponents/` để gọi file `.py` tương ứng để chạy pipeline hoàn chỉnh.
```
from english_grading.conponents import ModelTrain
class ModelTrain_Pipeline:
    def __init__(self, config):
        self.config = config
        self.modeltrain = Modeltrain(config=self.config)
    def main(self):
        self.modeltrain.data_ingestion()
        self.data_transformation()
        self.model_train()
```

#### ./utils/
Thì util thôi, mn thích lưu thì lưu, không thì thôi.

---
### main.py
Gọi từng pipeline và chạy
```
from english_grading.config import ConfigurationManager
from english_grading.logging import logger
from english_grading.pipeline import ModelTrain_Pipeline

def main():
    config_manager = ConfigurationManager(
        config_filepath=CONFIG_FILE_PATH
    )
    try:
        STAGE_NAME = stage_name("STAGE ?: MODEL TRAINING")
        logger.info(STAGE_NAME)
        model_train = ModelTrain_Pipeline(config=config_manager)
        model_train.main()
    except KeyboardInterrupt as k:
        logger.exception(k)
        # initializer.shutdown()
    except Exception as e:
        logger.exception(e)
        raise e
```

#### Notes
- Các thư mục trong `src/english_grading/*`, nếu tạo file gì thì trong `__init__.py` tương ứng nhớ `from .<tên file py> import *` để có thể sử dụng
- file requirements.txt cũng thêm tiền tố feature đang làm.