! snow17_shim.f90
!
! Thin bind(C) wrapper around EXSNOW19 (from external/snow17/src/snow19/,
! vendored upstream, Apache-2.0) that loops it over a time series for a
! single HRU with state threaded explicitly. See notes/NOTES.md for the
! state contract.
!
! Deliberately does NOT link against src/share, src/bmi, or src/driver —
! only the 13 plain FORTRAN-77 files in src/snow19/.
!
! Units: everything in mm/day. Caller must convert forcing to mm per
! timestep before calling (no mm/s <-> mm/step conversion happens here).
!
! Real kind: EXSNOW19 uses gfortran's default REAL (4 bytes). This file
! must never be built with -fdefault-real-8, or the argument association
! with the F77 code silently breaks.

module snow17_shim_mod
  use iso_c_binding, only: c_int, c_float
  implicit none

contains

  subroutine snow17_run(n, idt, idts, iyr, imn, ida,             &
                         pcp, tmp,                                &
                         alat, elev, scf, mfmax, mfmin, uadj, si, &
                         nmf, tipm, mbase, pxtemp, plwhc, daygm,  &
                         adc11,                                   &
                         cs_io, tprev_io,                         &
                         raim, sneqv, snowh) bind(C, name="snow17_run")

    ! -- series length and fixed timestep config (constant for the run) --
    integer(c_int), intent(in), value :: n        ! number of timesteps
    integer(c_int), intent(in), value :: idt       ! timestep, hours (24 = daily)
    integer(c_int), intent(in), value :: idts      ! timestep, seconds (86400 = daily)

    ! -- per-timestep calendar (avoids Gregorian/leap-year math in Fortran) --
    integer(c_int), intent(in) :: iyr(n), imn(n), ida(n)

    ! -- per-timestep forcing, mm/day and deg C --
    real(c_float), intent(in) :: pcp(n), tmp(n)

    ! -- basin attributes + learnable scalar parameters (single HRU) --
    real(c_float), intent(in), value :: alat, elev, scf, mfmax, mfmin, &
                                         uadj, si, nmf, tipm, mbase,    &
                                         pxtemp, plwhc, daygm
    real(c_float), intent(in) :: adc11(11)   ! areal depletion curve

    ! -- carryover state, threaded in/out; CS(19) is the full envelope, --
    ! -- TPREV is "previous timestep's air temperature", owned by the  --
    ! -- caller since neither PACK19 nor EXSNOW19 writes it back.      --
    real(c_float), intent(inout) :: cs_io(19)
    real(c_float), intent(inout) :: tprev_io

    ! -- outputs --
    real(c_float), intent(out) :: raim(n)   ! rain-plus-melt, mm/day -> feeds HBV
    real(c_float), intent(out) :: sneqv(n)  ! SWE, m (diagnostic)
    real(c_float), intent(out) :: snowh(n)  ! snow depth, m (diagnostic)

    external :: exsnow19

    real :: adc12(12)
    real :: pa
    real :: snow_out   ! SXFALL diagnostic (snowfall), not exposed
    integer :: t

    ! Surface pressure from elevation (Anderson 2006). Computed once here
    ! because EXSNOW19's own ELEV1 -> ELEV conversion feeds a PA formula
    ! that is commented out in exsnow19.f -- PA must come from outside.
    pa = 33.86 * (29.9 - 0.335*(elev/100.0) + 0.00022*((elev/100.0)**2.4))

    ! Work around the ADC(11) [exsnow19.f] vs ADC(12) [PACK19.f] dummy-array
    ! size mismatch: pad to 12 elements so PACK19's declared dummy size is
    ! never larger than the actual argument. No path we exercise (data
    ! assimilation is off: IUPWE=IUPSC=0 inside EXSNOW19) reads element 12,
    ! but this removes the size mismatch regardless of that.
    adc12(1:11) = adc11
    adc12(12) = 1.0

    do t = 1, n
      call exsnow19(idts, idt, ida(t), imn(t), iyr(t),                    &
                     pcp(t), tmp(t), raim(t), sneqv(t), snow_out, snowh(t), &
                     alat, scf, mfmax, mfmin, uadj, si, nmf, tipm,          &
                     mbase, pxtemp, plwhc, daygm, elev, pa, adc12,          &
                     cs_io, tprev_io)

      ! Caller-side bookkeeping: TPREV for the next call is this step's
      ! air temperature. Must happen AFTER the call, using this step's TMP.
      tprev_io = tmp(t)
    end do

  end subroutine snow17_run

end module snow17_shim_mod
