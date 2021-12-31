add_chopper_library(CHOPPERTypingCPA
  Algorithm.cpp
  Interfaces.cpp
  Types.cpp

  DEPENDS
  CHOPPERTypingCPAInterfacesIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR
)
