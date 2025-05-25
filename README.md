# FaceDangerTriangle 🔺

Détection visuelle du *triangle de la mort* sur le visage pour sensibiliser aux risques dermatologiques liés au toucher ou perçage de cette zone critique.

---

## Table des matières

* [Présentation](#présentation)
* [Fonctionnement](#fonctionnement)
* [Installation](#installation)
* [Usage](#usage)
* [Exemple](#exemple)
* [Applications](#applications)
* [Contributeurs](#contributeurs)
* [Licence](#licence)

---

## Présentation

Le *triangle de la mort* est une zone située entre le nez et la lèvre supérieure où percer un bouton peut entraîner de graves infections, pouvant se propager jusqu’au cerveau. Ce projet utilise OpenCV et MediaPipe pour détecter cette zone sur une image vidéo en temps réel, et avertir l’utilisateur lorsqu’une main s’en approche.

---

## Fonctionnement

* La détection de la zone critique est basée sur trois points clés du visage (landmarks MediaPipe).
* La zone est visualisée par un triangle vert par défaut.
* Lorsqu’une main est détectée dans cette zone, le triangle devient rouge, indiquant un potentiel danger.
* Ce système permet une sensibilisation interactive en temps réel.

---

## Installation

1. **Pré-requis** : Python 3.7 ou plus
2. Installer les dépendances via pip :

```bash
pip install opencv-python mediapipe numpy rich
```

3. Cloner ce dépôt :

```bash
git clone https://github.com/ArielShadrac/FaceDangerTriangle.git
cd FaceDangerTriangle
```

4. Lancer le script principal :

```bash
python FaceDnageTriangle.py
```

---

## Usage

* Lancez la webcam avec le script.
* Le triangle de la mort est affiché sur votre visage (entre nez et lèvre supérieure).
* Lorsque votre main s’approche, la couleur du triangle passe au rouge et un avertissement s’affiche à l’écran.

---

## Applications

* Sensibilisation dermatologique grand public
* Formation médicale et paramédicale
* Assistance en chirurgie esthétique ou dermatologique
* Education aux bonnes pratiques d’hygiène cutanée
* Projet à potentiel fort dans les pays en développement

---

## Contributeurs

* F Ariel Shadrac OUEDRAOGO – Développeur principal


---

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

---
