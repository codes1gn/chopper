use std::fmt;

#[derive(Debug, Clone)]
pub struct EmptyCmdBufferError;

#[derive(Debug, Clone)]
pub struct RuntimeStatusError;

impl fmt::Display for EmptyCmdBufferError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Command Buffer is end, but not terminate the inst fetch!"
        )
    }
}

impl fmt::Display for RuntimeStatusError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Runtime execution status is not Ok!")
    }
}
