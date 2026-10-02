# BioSense+ ImageLab

[Português](#português) | [English](#english)

---

# Português

## Sobre o projeto

**BioSense+ ImageLab** é uma ferramenta computacional para análise biomecânica da marcha utilizando visão computacional.

O sistema processa vídeos de movimento humano, identifica pontos anatômicos por meio de estimativa de pose e utiliza essas informações para calcular parâmetros relacionados à marcha, como ângulos articulares, fases de apoio e balanço, eventos de contato, passos e passadas.

O projeto faz parte da plataforma **BioSense+** e também prevê a integração dos dados obtidos por vídeo com informações provenientes de uma plataforma de pressão plantar baseada em uma matriz piezoresistiva.

O desenvolvimento é realizado no contexto de um Trabalho de Conclusão de Curso em Engenharia da Computação.

---

## Objetivos

O ImageLab foi desenvolvido com o objetivo de:

- processar vídeos de marcha humana;
- identificar e acompanhar pontos anatômicos;
- calcular ângulos articulares;
- realizar calibração espacial de pixels para centímetros;
- identificar fases de apoio e balanço;
- detectar eventos como contato inicial e retirada do pé;
- estimar comprimento de passos e passadas;
- permitir correção manual dos pontos detectados;
- exportar dados biomecânicos para análises posteriores;
- integrar futuramente os dados de vídeo com pressão plantar;
- contribuir para a construção de um banco de dados biomecânico.

---

## Principais funcionalidades

### Estimativa de pose

O sistema utiliza **MediaPipe Pose / BlazePose** para detectar 33 pontos corporais em cada quadro do vídeo.

Entre os pontos utilizados na análise estão:

- ombros;
- quadris;
- joelhos;
- tornozelos;
- calcanhares;
- extremidades dos pés.

Os pontos detectados são acompanhados ao longo do vídeo e utilizados para gerar séries temporais do movimento.

### Estabilização dos membros inferiores

Foi implementado um tratamento adicional para reduzir inversões entre os lados esquerdo e direito durante cruzamentos das pernas ou perdas temporárias de detecção.

O algoritmo compara a continuidade das trajetórias dos membros entre quadros consecutivos e pode corrigir automaticamente a identificação dos lados.

### Correção manual e rastreamento

Quando um ponto anatômico não é identificado corretamente, o usuário pode ajustar manualmente sua posição.

Após a correção, o sistema tenta acompanhar o ponto nos quadros seguintes utilizando:

- rastreador CSRT do OpenCV;
- segmentação local no espaço de cores HSV;
- operações morfológicas;
- análise de componentes conectados;
- propagação temporal da correção quando necessário.

### Calibração espacial

Uma referência de comprimento conhecido pode ser utilizada para estabelecer a relação entre pixels e centímetros.

A partir da calibração, o sistema pode calcular medidas físicas como:

- deslocamentos;
- distância entre pontos;
- comprimento do passo;
- comprimento da passada.

### Análise angular

Os ângulos são calculados a partir das coordenadas dos landmarks.

Na vista lateral são analisadas principalmente as articulações de:

- quadril;
- joelho;
- tornozelo.

Também existem medidas auxiliares para análises frontais, incluindo inclinação da pelve, inclinação do tronco e parâmetros relacionados ao alinhamento dos membros inferiores.

### Análise da marcha

O ImageLab utiliza a posição e o deslocamento dos pontos associados aos pés para estimar os estados de:

- **APOIO**;
- **BALANÇO**.

A partir das transições entre esses estados, são identificados eventos como:

- contato inicial;
- retirada do pé.

Esses eventos são posteriormente utilizados para organizar passos e passadas dos membros esquerdo e direito.

### Exportação dos resultados

O processamento gera:

- vídeo MP4 com informações biomecânicas sobrepostas;
- arquivo CSV contendo os valores calculados para cada quadro.

Entre os dados exportados podem estar:

- tempo;
- ângulos articulares;
- fases da marcha;
- eventos;
- passos;
- passadas;
- distâncias entre landmarks;
- outras variáveis biomecânicas calculadas pelo sistema.

---

## Pressão plantar

O projeto BioSense+ também inclui uma plataforma plantar baseada em uma matriz piezoresistiva de **16 × 16 posições**, totalizando 256 regiões de sensoriamento.

O elemento sensível utilizado na plataforma é baseado em **Velostat**.

A integração sincronizada entre os dados da plataforma plantar e o ImageLab encontra-se em desenvolvimento.

O objetivo dessa integração é relacionar informações obtidas pela câmera, como contato inicial e retirada do pé, com mudanças observadas na distribuição da pressão plantar.

---

## Tecnologias utilizadas

- Python 3.12
- OpenCV / OpenCV-Contrib 4.11
- MediaPipe 0.10.21
- NumPy 1.26.4
- Pillow 10.4.0
- Tkinter / ttk

---

## Estrutura do projeto

```text
marcha_pose_mvp/
│
├── app.py
├── processor.py
├── reviewer.py
├── i18n.py
├── test_processor.py
│
├── locales/
│   ├── pt.json
│   ├── en.json
│   └── es.json
│
└── README.md
```

### Arquivos principais

**`app.py`**  
Responsável pela inicialização da aplicação e da interface gráfica.

**`processor.py`**  
Contém as principais rotinas de processamento de vídeo, estimativa de pose, cálculos biomecânicos e análise da marcha.

**`reviewer.py`**  
Responsável pela revisão dos resultados, correção manual dos landmarks e rastreamento dos pontos corrigidos.

**`i18n.py`**  
Gerencia os recursos de internacionalização da interface.

**`locales/`**  
Contém os arquivos de tradução da aplicação.

**`test_processor.py`**  
Contém testes relacionados às rotinas de processamento.

---

## Instalação

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Crie um ambiente virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install mediapipe==0.10.21
pip install opencv-contrib-python==4.11.0.86
pip install numpy==1.26.4
pip install Pillow==10.4.0
```

---

## Execução

Com o ambiente virtual ativado:

```bash
python app.py
```

A interface gráfica será aberta e permitirá selecionar um vídeo para processamento e análise.

---

## Fluxo simplificado

```text
Vídeo de entrada
      │
      ▼
Leitura dos quadros
      │
      ▼
MediaPipe Pose
      │
      ▼
Detecção dos landmarks
      │
      ▼
Estabilização temporal
      │
      ▼
Correção / revisão manual
      │
      ▼
Calibração espacial
      │
      ▼
Cálculo biomecânico
      │
      ├── Ângulos
      ├── Apoio / balanço
      ├── Eventos
      ├── Passos
      └── Passadas
      │
      ▼
Exportação
      │
      ├── MP4
      └── CSV
```

---

## Status do projeto

O projeto encontra-se em desenvolvimento acadêmico.

Atualmente estão implementados:

- processamento de vídeos;
- estimativa de pose;
- visualização dos landmarks;
- correção manual;
- rastreamento dos pontos corrigidos;
- calibração espacial;
- cálculo de ângulos;
- identificação de fases da marcha;
- identificação de eventos;
- cálculo de passos e passadas;
- exportação de vídeo e dados.

Entre as próximas etapas estão:

- validação quantitativa das medidas;
- integração sincronizada com a plataforma plantar;
- ampliação dos testes experimentais;
- estruturação do banco de dados biomecânico.

---

## Aviso

O BioSense+ ImageLab é um projeto acadêmico e de pesquisa.

Os resultados obtidos pelo software ainda estão em processo de validação e **não devem ser utilizados como diagnóstico médico ou como substitutos de equipamentos clínicos ou laboratoriais certificados**.

---

# English

## About the project

**BioSense+ ImageLab** is a computer vision-based tool developed for biomechanical gait analysis.

The system processes human movement videos, detects anatomical landmarks using pose estimation, and uses this information to calculate gait-related parameters such as joint angles, stance and swing phases, gait events, steps, and strides.

The project is part of the **BioSense+** platform and also aims to integrate video-based measurements with data acquired from a piezoresistive plantar pressure platform.

The software is being developed as part of a Computer Engineering undergraduate research and final-year project.

---

## Objectives

ImageLab was developed to:

- process human gait videos;
- detect and track anatomical landmarks;
- calculate joint angles;
- perform spatial calibration from pixels to centimeters;
- identify stance and swing phases;
- detect events such as initial contact and foot-off;
- estimate step and stride lengths;
- allow manual correction of detected landmarks;
- export biomechanical data for further analysis;
- integrate video data with plantar pressure information;
- contribute to the development of a biomechanical database.

---

## Main features

### Pose estimation

The system uses **MediaPipe Pose / BlazePose** to detect 33 body landmarks in each video frame.

Landmarks used in the analysis include:

- shoulders;
- hips;
- knees;
- ankles;
- heels;
- foot tips.

Detected landmarks are tracked throughout the video and used to create temporal movement data.

### Lower-limb stabilization

An additional stabilization method was implemented to reduce incorrect left/right limb switching during leg crossings or temporary detection failures.

The algorithm evaluates trajectory continuity between consecutive frames and can automatically correct side identification.

### Manual correction and tracking

When an anatomical landmark is incorrectly detected, the user can manually adjust its position.

After correction, the system attempts to track the new position in subsequent frames using:

- OpenCV CSRT tracker;
- local HSV-based segmentation;
- morphological operations;
- connected-component analysis;
- temporal correction propagation when required.

### Spatial calibration

A reference object with a known physical length can be used to determine the relationship between pixels and centimeters.

After calibration, the software can calculate physical measurements such as:

- displacement;
- distance between landmarks;
- step length;
- stride length.

### Joint-angle analysis

Joint angles are calculated from landmark coordinates.

The main joints analyzed in the lateral view are:

- hip;
- knee;
- ankle.

Additional measurements are also available for frontal-view analysis, including pelvic tilt, trunk lean, and lower-limb alignment parameters.

### Gait analysis

ImageLab analyzes the position and displacement of foot landmarks to estimate:

- **STANCE**;
- **SWING**.

Transitions between these states are used to identify events such as:

- initial contact;
- foot-off.

These events are then used to organize left and right steps and strides.

### Result export

The processing pipeline generates:

- an MP4 video containing biomechanical overlays;
- a CSV file containing calculated values for each processed frame.

Exported data may include:

- time;
- joint angles;
- gait phases;
- gait events;
- steps;
- strides;
- distances between landmarks;
- additional biomechanical measurements.

---

## Plantar pressure

The BioSense+ project also includes a plantar pressure platform based on a **16 × 16 piezoresistive matrix**, providing 256 sensing regions.

The sensing element is based on **Velostat**.

Synchronized integration between the plantar pressure platform and ImageLab is currently under development.

The objective is to associate camera-based events, such as initial contact and foot-off, with changes in plantar pressure distribution.

---

## Technologies

- Python 3.12
- OpenCV / OpenCV-Contrib 4.11
- MediaPipe 0.10.21
- NumPy 1.26.4
- Pillow 10.4.0
- Tkinter / ttk

---

## Project structure

```text
marcha_pose_mvp/
│
├── app.py
├── processor.py
├── reviewer.py
├── i18n.py
├── test_processor.py
│
├── locales/
│   ├── pt.json
│   ├── en.json
│   └── es.json
│
└── README.md
```

### Main files

**`app.py`**  
Application entry point and graphical user interface.

**`processor.py`**  
Contains the main video-processing, pose-estimation, biomechanical calculation, and gait-analysis routines.

**`reviewer.py`**  
Handles result review, manual landmark correction, and corrected-point tracking.

**`i18n.py`**  
Manages interface internationalization.

**`locales/`**  
Contains interface translation files.

**`test_processor.py`**  
Contains tests related to the processing routines.

---

## Installation

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install mediapipe==0.10.21
pip install opencv-contrib-python==4.11.0.86
pip install numpy==1.26.4
pip install Pillow==10.4.0
```

---

## Running the application

With the virtual environment activated:

```bash
python app.py
```

The graphical interface will open and allow the user to select a video for processing and analysis.

---

## Simplified pipeline

```text
Input video
     │
     ▼
Frame acquisition
     │
     ▼
MediaPipe Pose
     │
     ▼
Landmark detection
     │
     ▼
Temporal stabilization
     │
     ▼
Manual review / correction
     │
     ▼
Spatial calibration
     │
     ▼
Biomechanical analysis
     │
     ├── Joint angles
     ├── Stance / swing
     ├── Gait events
     ├── Steps
     └── Strides
     │
     ▼
Export
     │
     ├── MP4
     └── CSV
```

---

## Project status

The project is currently under academic development.

Implemented features include:

- video processing;
- pose estimation;
- landmark visualization;
- manual landmark correction;
- corrected-point tracking;
- spatial calibration;
- joint-angle calculation;
- gait-phase estimation;
- gait-event detection;
- step and stride calculation;
- video and data export.

Planned next steps include:

- quantitative validation of the measurements;
- synchronized integration with the plantar pressure platform;
- expansion of the experimental evaluation;
- development of the biomechanical database.

---

## Disclaimer

BioSense+ ImageLab is an academic research project.

The measurements produced by the software are still undergoing validation and **should not be used for medical diagnosis or as a replacement for certified clinical or laboratory equipment**.
