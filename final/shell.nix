with import <nixpkgs> {};
  mkShell {
    buildInputs = [
      opencv
      pkg-config
      cppcheck
    ];
  }
