{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/23.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix-flake.url = "github:nix-community/poetry2nix";
    typix = {
      url = "github:loqusion/typix";
      inputs.nixpkgs.follows = "nixpkgs-unstable";
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      poetry2nix-flake,
      typix,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (pkgs) lib;

        poetry2nix = poetry2nix-flake.lib.mkPoetry2Nix { inherit pkgs; };

        inherit (poetry2nix)
          mkPoetryApplication
          mkPoetryEnv
          ;

        mnist-digits-nb-nn = mkPoetryApplication {
          projectDir = ./.;
          preferWheels = true;
        };

        mnist-digits-nb-nn-env = mkPoetryEnv {
          projectDir = ./.;
          preferWheels = true;
          editablePackageSources = {
            mnist-digits-nb-nn = ./mnist_digits_nb_nn;
          };
          extraPackages =
            ps: with ps; [
              tornado

              python-lsp-server
              mypy
              python-lsp-black
              pyls-isort
              pylsp-mypy
              flake8
              flake8-bugbear
              rope
            ];
        };

        typixLib = typix.lib.${system};

        reportBuildArgs = {
          typstSource = "report.typ";
          src = lib.fileset.toSource {
            root = ./report;
            fileset = lib.fileset.unions [
              (lib.fileset.fromSource (typixLib.cleanTypstSource ./report))
              ./report/graphics
              ./report/data
              ./report/references.bib
            ];
          };
        };

        report = typixLib.buildTypstProject reportBuildArgs;
        build-report = typixLib.buildTypstProjectLocal reportBuildArgs;
        watch-report = typixLib.watchTypstProject { typstSource = "report/report.typ"; };
      in
      {
        devShell = typixLib.devShell {
          packages = [
            pkgs.poetry
            mnist-digits-nb-nn-env
            pkgs.cudatoolkit
            pkgs.cudaPackages.cudnn

            build-report
            watch-report
          ];

          MPLBACKEND = "WebAgg";

          # CUDA: https://discourse.nixos.org/t/installing-pytorch-into-a-virtual-python-environment/34720/2
          LD_LIBRARY_PATH = "${
            pkgs.lib.makeLibraryPath (with pkgs; [ libz ])
          }:${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH";

          CUDA_PATH = "${pkgs.cudatoolkit}";
        };

        checks = {
          inherit
            mnist-digits-nb-nn
            report
            build-report
            watch-report
            ;
        };

        packages = {
          inherit
            mnist-digits-nb-nn
            report
            ;

          default = mnist-digits-nb-nn;
        };

        apps =
          let
            mnist-digits-nb-nn-app = flake-utils.lib.mkApp {
              drv = self.packages.${system}.default;
            };
          in
          {
            default = mnist-digits-nb-nn-app;
            mnist-digits-nb-nn = mnist-digits-nb-nn-app;

            build-report = flake-utils.lib.mkApp {
              drv = build-report;
            };

            watch-report = flake-utils.lib.mkApp {
              drv = watch-report;
            };
          };
      }
    );
}
