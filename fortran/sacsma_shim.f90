! sacsma_shim.f90
!
! Thin bind(C) wrapper around EXSAC (from external/sac-sma/src/sac/,
! vendored upstream, Apache-2.0, patched at build time -- see
! patches/sac1_bypass_ratio_check_save_fix.patch and notes/NOTES.md) that
! loops it over a time series for a single HRU with state threaded
! explicitly. Mirrors snow17_shim.f90's design; see that file and
! CLAUDE.md for the shared rationale.
!
! Unlike EXSNOW19, EXSAC's state is 6 individually-named INOUT arguments
! (UZTWC, UZFWC, LZTWC, LZFSC, LZFPC, ADIMC), not a packed array -- no
! NEXLAG-style bookkeeping needed. state_io(6) here is just those six in
! that fixed order, purely for a uniform calling convention with the
! snow17 shim's cs_io.
!
! Real kind: EXSAC/SAC1 use explicit DOUBLE PRECISION throughout (unlike
! Snow17's default 4-byte REAL) -- confirmed by reading ex_sac1.f90,
! sac1.f90, and sac_data_mod.f90 (dp = SELECTED_REAL_KIND(15, 307)).
! This shim uses c_double to match. Do NOT use c_float here.
!
! Units: PCP and ETP are mm per timestep (depth), matching the shim's
! overall mm/day convention -- caller converts from mm/s rate forcing
! before calling, same as snow17_shim.f90. TMP is accepted for interface
! completeness (mirrors EXSAC's own signature) but currently has zero
! effect on output: EXSAC hardcodes IFRZE=0, and TMP is only read inside
! the frozen-ground path (FROST1), which that flag disables entirely.

module sacsma_shim_mod
  use iso_c_binding, only: c_int, c_double
  implicit none

contains

  subroutine sacsma_run(n, dtm,                                       &
                         pcp, tmp, etp,                                &
                         uztwm, uzfwm, uzk, pctim, adimp, riva, zperc, &
                         rexp, lztwm, lzfsm, lzfpm, lzsk, lzpk, pfree, &
                         side, rserv,                                  &
                         state_io,                                     &
                         qs, qg, q, eta, roimp, sdro, ssur, sif,      &
                         bfs, bfp, bfncc) bind(C, name="sacsma_run")

    ! -- series length and fixed timestep (constant for the run) --
    integer(c_int), intent(in), value :: n     ! number of timesteps
    real(c_double), intent(in), value :: dtm   ! timestep, seconds (86400 = daily)

    ! -- per-timestep forcing: mm/day, deg C, mm/day --
    real(c_double), intent(in) :: pcp(n), tmp(n), etp(n)

    ! -- learnable scalar parameters (single HRU), fixed for the run --
    real(c_double), intent(in), value :: uztwm, uzfwm, uzk, pctim, adimp, riva, zperc, &
                                          rexp, lztwm, lzfsm, lzfpm, lzsk, lzpk, pfree, &
                                          side, rserv

    ! -- carryover state, threaded in/out: UZTWC, UZFWC, LZTWC, LZFSC, LZFPC, ADIMC --
    real(c_double), intent(inout) :: state_io(6)

    ! -- outputs, mm/day unless noted --
    real(c_double), intent(out) :: qs(n)     ! surface flow (ROIMP+SDRO+SSUR+SIF)
    real(c_double), intent(out) :: qg(n)     ! groundwater flow (BFS+BFP)
    real(c_double), intent(out) :: q(n)      ! TCI, total channel inflow -> "runoff" for the NSE loss
    real(c_double), intent(out) :: eta(n)    ! actual evapotranspiration
    real(c_double), intent(out) :: roimp(n)  ! runoff, minimum impervious area
    real(c_double), intent(out) :: sdro(n)   ! direct runoff, ADIMP area
    real(c_double), intent(out) :: ssur(n)   ! surface runoff
    real(c_double), intent(out) :: sif(n)    ! interflow
    real(c_double), intent(out) :: bfs(n)    ! baseflow, secondary
    real(c_double), intent(out) :: bfp(n)    ! baseflow, primary
    real(c_double), intent(out) :: bfncc(n)  ! baseflow, non-channel component (a real loss term -- see mass balance)

    external :: exsac

    integer :: t
    integer :: nsold

    ! "NSOLD, which isn't used" -- runSac.f90's own comment on this argument.
    nsold = 1

    do t = 1, n
      call exsac(nsold, dtm, pcp(t), tmp(t), etp(t),                    &
                 uztwm, uzfwm, uzk, pctim, adimp, riva, zperc,          &
                 rexp, lztwm, lzfsm, lzfpm, lzsk, lzpk, pfree,          &
                 side, rserv,                                           &
                 state_io(1), state_io(2), state_io(3),                 &
                 state_io(4), state_io(5), state_io(6),                 &
                 qs(t), qg(t), q(t), eta(t), roimp(t), sdro(t),         &
                 ssur(t), sif(t), bfs(t), bfp(t), bfncc(t))
    end do

  end subroutine sacsma_run

end module sacsma_shim_mod
