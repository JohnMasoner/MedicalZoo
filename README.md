<div id="top"></div>

# A General Medical Image Segmentation Model in PyTorch
<!-- PROJECT LOGO -->

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#recent-updates">Recent Updates</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

There are many medical image segmentation models out there, but few frameworks focus on mono medical image segmentation or focus on both. This project is a general medical image segmentation model in PyTorch, and to be more precise, it is a general medical image segmentation model in PyTorch that is not only able to segment medical images, but also can be used for other tasks.

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have contributed to expanding this template!


<p align="right">(<a href="#top">back to top</a>)</p>

<!--Recent Updates-->
## Recent Updates

* 2022-04-29: support multi-modal medical image segmentation and adjacent layer.
* 2022-04-26: support multi-classes and upload inference code.
* 2022-04-24: update multi-clases metrics.
* 2022-04-07: optimized logger code. Mean the metrics.
* 2022-04-06: Added Transforms , crop data by label(over-sampling) and Generate patches mask
* 2022-04-02: Added some script. examples, save data.
* 2022-04-02: Mean the metrics
* 2022-03-30: Added 3dDataloader
* 2022-03-25: Added Configuration README
* 2022-03-24: Added License section and README.md
* 2022-03-11: Fixed Release Notes
* 2022-03-10: Initial Release

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

<!-- Installation -->
### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo

   ```sh
   git clone https://github.com/JohnMasoner/MedicalZoo.git
   ```

2. Install Python packages

   ```sh
   pip install -r requirements.txt
   ```

3. Initialization Configuration.
you could apply change configuration to your datasets and customize your training strategies.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

```python
python run.py [configuration] [training_model]
```

_For more information about configuration, Please [Config ReadMe](https://github.com/JohnMasoner/MedicalZoo/tree/main/config)_

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

* [ ] Add a blog to introduce the project
* [x] Add More Modules to better training your model(More TODO!)
* [x] Add 3D Modeling
* [x] Add more data augmentation
* [ ] Multi-language Support
  * [ ] Chinese
  * [ ] Chinese (Traditional)
  * [ ] French

See the [open issues](https://github.com/JohnMasoner/MedicalZoo/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Mason Ma - masoner6429@gmail.com


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* [Pytorch-Medical-Segmentation](https://github.com/MontaEllis/Pytorch-Medical-Segmentation)
* [MONAI](https://github.com/Project-MONAI/MONAI)

<p align="right">(<a href="#top">back to top</a>)</p>
