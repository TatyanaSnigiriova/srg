import unittest
import super_gradients
from super_gradients.training import MultiGPUMode, models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
import os
import shutil


class PretrainedModelsUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.imagenet_pretrained_models = ["resnet50", "repvgg_a0", "regnetY800"]

    def test_pretrained_resnet50_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet50_unit_test", multi_gpu=MultiGPUMode.OFF)
        model = models.get("resnet50", pretrained_weights="imagenet")
        trainer.test(model=model, test_loader=classification_test_dataloader(), test_metrics_list=[Accuracy()], metrics_progress_verbose=True)

    def test_pretrained_regnetY800_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY800_unit_test", multi_gpu=MultiGPUMode.OFF)
        model = models.get("regnetY800", pretrained_weights="imagenet")
        trainer.test(model=model, test_loader=classification_test_dataloader(), test_metrics_list=[Accuracy()], metrics_progress_verbose=True)

    def test_pretrained_repvgg_a0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_repvgg_a0_unit_test", multi_gpu=MultiGPUMode.OFF)
        model = models.get("repvgg_a0", pretrained_weights="imagenet", arch_params={"build_residual_branches": True})
        trainer.test(model=model, test_loader=classification_test_dataloader(), test_metrics_list=[Accuracy()], metrics_progress_verbose=True)

    def tearDown(self) -> None:
        if os.path.exists("~/.cache/torch/hub/"):
            shutil.rmtree("~/.cache/torch/hub/")


if __name__ == "__main__":
    unittest.main()
