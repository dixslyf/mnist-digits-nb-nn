{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/23.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix-flake.url = "github:nix-community/poetry2nix";
  };
  outputs = { nixpkgs, nixpkgs-unstable, flake-utils, poetry2nix-flake, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      unstable-pkgs = nixpkgs-unstable.legacyPackages.${system};

      poetry2nix = poetry2nix-flake.lib.mkPoetry2Nix { inherit pkgs; };

      inherit (poetry2nix) mkPoetryEnv;

      overrides = poetry2nix.overrides.withDefaults
        (_: prev: {
          gym-notices = prev.gym-notices.overridePythonAttrs (
            old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ prev.setuptools ];
            }
          );
        });

      env = mkPoetryEnv {
        projectDir = ./.;
        inherit overrides;
        preferWheels = true;
        extraPackages = ps: with ps; [
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
    in
    {
      devShell = pkgs.mkShell {
        packages = [
          unstable-pkgs.typst
          pkgs.poetry
          env
          pkgs.cudatoolkit
          pkgs.cudaPackages.cudnn
        ];

        MPLBACKEND = "WebAgg";

        # CUDA: https://discourse.nixos.org/t/installing-pytorch-into-a-virtual-python-environment/34720/2
        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath (with pkgs; [libz])}:${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH";

        CUDA_PATH = "${pkgs.cudatoolkit}";
      };
    });
}
