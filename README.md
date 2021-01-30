# Sorbonne University Paris: Master 2 ISI

## Project: Estimating a person's age from the image of their face

@authors : **Sylvain XIA, Yuesheng ZHANG, Longyu ZHANG, Ronan RAPHAEL**

__Vous trouverez dans le repértoire Notebooks__ :  les notebooks que nous avons utilisés pour extraire les données(images et labels) / réaliser la partie recentrage des images / l'apprentissage de nos modèles.

   + _"Data_extract.ipynb"_ : utilisé pour extraire les données contenus dans la banque d'image et les enregistres dans les fichiers xtrain.npy (apprentissage) et xval.npy (validation).

   + _"label_extract.ipynb"_ : utilisé pour lire les labels de chaque image et selon un dictionnaire d'age, associe le label correspondant. Enregistre ces labels dans une fichier label_fair_face.npz, ytrain et yval sont séparés.

   + _"random_data_extract.ipynb"_ : extrait de manière aléatoire Nx1000 images dans la base d'apprentissage, construit xtrain et ytrain pour les images choisis et les sauvegarde dans un fichier data_randNk.npz

   + _"padding_125to025.ipynb"_ : contient la partie traitement d'image qui permet de centrer les images non centrées (padding=1.25). Dans ce notebook, nous ne regénérons pas d'images, nous montrons juste une méthode permettant de passer des images de padding=1.25 à des images centrées. Nous affichons quelques résultats sur la base qui a été utilisé pour l'apprentissage. L'apprentissage des modèles a été effectuée sur les images déjà centrées dans la base (padding=0.25) fournie https://github.com/joojs/fairface.

   + _"vgg16_reseaux.ipynb"_ : réseau inspiré de VGG16, utilisé pour l'apprentissage (l'apprentissage a été réalisé sur la base de validation padding=0.25 de fairface)
  
   + _"cnn_reseaux.ipynb"_ : réseau inspiré de https://github.com/rodrigobressan/keras-multi-output-model-utk-face, utilisé pour l'apprentissage (l'apprentissage a été réalisé sur la base de validation padding=0.25 de fairface)
 
  
__Vous trouverez dans le repértoire Data :__

   + _"label_fair_face.npz"_ : fichier contenant les labels ytrain et yval.

   + _"haarcascade_frontalface_default.xml"_ : fichier contenant la cascade de classifieur pré-entrainé de openCV. Utilisé pour la partie recentrage de l'image.
   
   Vous devez ajouter dans ce dossier la base de donnée extraite utilisé pour l'apprentissage [xval.npy](https://dropsu.sorbonne-universite.fr/s/fL447p68ei37Dw2), et un zip contenant les modèles appris par le réseau [saved_model.zip](https://dropsu.sorbonne-universite.fr/s/JyAxR7dGsJGgqoA)

__Vous trouverez dans le répertoire démo :__

  + _"padding_125to025_functions.py"_ : script qui regroupe les fonctions crées dans le notebook. 
  
  + _"load_and_predict.ipynb"_ : notebook permettant de lancer le programme final. Il charge 5 images tests de padding=1.25, les recentre à l'aide de la méthode montrer dans le notebook _"padding_125to025.ipynb"_, charge les deux modèles au et réalise l'estimation de l'âge avec les predicteurs de ceux ci.

  + _"img_test"_ : contient les images utilisées pour la démo
  
## 1. Requirements:

- Python 3.7
- NumPy (v1.19.2)
- matplotlib (v3.3.2)
- tensorflow (v2.3.1)
- openCV (v4.0.1+)

## 2. Instructions : 

Vous pouvez executer une démo de notre code en executant le fichier load_and_predict.ipynb dans le dossier demo. 
Pour que ce programme fonctionne il faut copier le contenu du dossier Data dans le dossier demo, télécharger les modèles appris par le réseau ici [saved_model.zip](https://dropsu.sorbonne-universite.fr/s/JyAxR7dGsJGgqoA) et les mettre dans le même dossier

